"""Regenerate device-portable source from hash-verified, pinned author files."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    upstream = ROOT / 'upstream'
    manifest = json.loads((upstream / 'manifest.json').read_text(encoding='utf-8'))
    generated = []
    for item in manifest['files']:
        source = upstream / item['path']
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError(f'Upstream source changed: {item["path"]}')
        if item['path'] == 'LICENSE':
            continue
        target = item['path'].replace('NPP_Codec/ImageCompression/', 'imagecompression/')
        target = target.replace('NPP_Codec/models/', 'npp_fixed/').replace('NPP_Codec_Variable/models/', 'npp_adaptive/')
        text = data.decode('utf-8')
        patches = []
        if target == 'imagecompression/model.py':
            text = text.replace('from models import *', 'from .models import *')
            text = text.replace('torch.zeros(input_image.size(0)', 'input_image.new_zeros(input_image.size(0)')
            text = text.replace('        print(total_bits_feature)', '        # Per-forward debug output removed.')
            patches = ['package-relative imports', 'noise follows input device/dtype', 'remove debug print']
        if target == 'imagecompression/models/context_module.py':
            text = text.replace('import torch\n', 'import torch\nimport torch.nn.functional as F\n')
            original = ('    def __call__(self, x):\n'
                        '        self.weight.data = self.weight.data * self.mask\n'
                        '        return super(MaskedConvolution2D, self).forward(x)')
            replacement = ('    def forward(self, x):\n'
                           '        return F.conv2d(x, self.weight * self.mask, self.bias, self.stride,\n'
                           '                        self.padding, self.dilation, self.groups)')
            if original not in text:
                raise ValueError('Upstream causal convolution changed')
            text = text.replace(original, replacement)
            text = '\n'.join(line for line in text.split('\n') if not line.strip().startswith(('print("%s %s MASKED', 'print(pre_mask[')))
            patches = ['functional causal mask instead of .data mutation', 'remove mask debug print']
        destination = ROOT / 'npp_proxy/vendor' / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding='utf-8', newline='\n')
        generated.append({'source': item['path'], 'destination': str(destination.relative_to(ROOT)),
                          'patches': patches, 'sha256': hashlib.sha256(destination.read_bytes()).hexdigest()})
    for folder in ['vendor', 'vendor/imagecompression', 'vendor/npp_fixed', 'vendor/npp_adaptive']:
        (ROOT / 'npp_proxy' / folder / '__init__.py').write_text('', encoding='utf-8')
    (ROOT / 'upstream/port_manifest.json').write_text(json.dumps(generated, indent=2), encoding='utf-8')
    print(f'Generated {len(generated)} native files from {manifest["commit"]}')


if __name__ == '__main__':
    main()
