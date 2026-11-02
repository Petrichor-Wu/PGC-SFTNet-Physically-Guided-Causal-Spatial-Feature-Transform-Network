import torch
import torch.nn as nn
import torch.nn.functional as F
import json

# Variable aliases.
NAME_MAPPING = {
    'WindSpeed': 'Wind',
    'WINDSPEED': 'Wind',
    'TEMP': 'T2m',
    'Temperature': 'T2m',
    'SoilMoisture': 'SM',
    'DEM': 'DEM',
    'NDVI': 'NDVI',
    'LST': 'LST',
    'Albedo': 'Albedo',
    'Slope': 'Slope',
    'Aspect': 'Aspect',
    'T2m': 'T2m'
}


class SFTLayer(nn.Module):
    """Spatial feature transform."""
    def __init__(self, in_channels, cond_channels):
        super().__init__()
        self.predict_scale = nn.Conv2d(cond_channels, in_channels, 1)
        self.predict_shift = nn.Conv2d(cond_channels, in_channels, 1)

        # Initialize as an identity transform.
        nn.init.constant_(self.predict_scale.weight, 0)
        nn.init.constant_(self.predict_scale.bias, 0)
        nn.init.constant_(self.predict_shift.weight, 0)
        nn.init.constant_(self.predict_shift.bias, 0)

    def forward(self, x, condition):
        # Match spatial dimensions.
        if condition.shape[-2:] != x.shape[-2:]:
            condition = F.interpolate(condition, size=x.shape[-2:], mode='bilinear', align_corners=False)

        scale = self.predict_scale(condition)
        shift = self.predict_shift(condition)

        # Limit scale modulation to forty percent.
        scale = torch.tanh(scale) * 0.4

        return x * (1 + scale) + shift


class SELayer(nn.Module):
    """Channel attention with global pooling."""
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class CausalResBlock(nn.Module):
    def __init__(self, channels, sft_groups, use_se=True):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

        self.sft_layers = nn.ModuleDict()
        # Skip empty conditioning groups.
        for group_name, ch_dim in sft_groups.items():
            if ch_dim > 0:
                self.sft_layers[group_name] = SFTLayer(channels, ch_dim)

        self.use_se = use_se
        if self.use_se:
            self.se = SELayer(channels)

    def forward(self, x, conditions):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)

        # Apply L3 before L2.
        if 'L3' in self.sft_layers and 'L3' in conditions:
            out = self.sft_layers['L3'](out, conditions['L3'])

        if 'L2' in self.sft_layers and 'L2' in conditions:
            out = self.sft_layers['L2'](out, conditions['L2'])

        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_se:
            out = self.se(out)

        return out + residual


class CIG_SFTNet(nn.Module):
    def __init__(self, pcmci_json_path, feature_dim=64, num_blocks=8):
        super().__init__()

        # T2m is the target, not an input.
        self.var_channels = {
            'LST': 1, 'NDVI': 1, 'Albedo': 1, 'SM': 1, 'Wind': 1,
            'DEM': 1, 'Slope': 1, 'Aspect': 1
        }

        self.causal_structure = self._parse_pcmci_json(pcmci_json_path)
        print("Causal groups:", self.causal_structure)

        # Count conditioning channels.
        self.sft_config = {}
        for group in ['L2', 'L3']:
            channels = 0
            valid_vars = []
            for v in self.causal_structure[group]:
                if v in self.var_channels:
                    channels += self.var_channels[v]
                    valid_vars.append(v)
                else:
                    pass

            self.sft_config[group] = channels
            self.causal_structure[group] = valid_vars

        print("SFT channels:", self.sft_config)

        total_input_ch = sum(self.var_channels.values())
        self.head = nn.Conv2d(total_input_ch, feature_dim, 3, padding=1)

        self.body = nn.ModuleList([
            CausalResBlock(feature_dim, self.sft_config, use_se=True) for _ in range(num_blocks)
        ])

        self.tail = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_dim, 1, 1)
        )

    def _parse_pcmci_json(self, json_path):
        """Load causal groups and apply fixed priors."""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        def clean_name(name_list):
            cleaned = []
            for n in name_list:
                # Normalize variable names.
                n_clean = n.replace('DAILY_', '').replace('_average', '')
                n_std = NAME_MAPPING.get(n_clean, n_clean)
                cleaned.append(n_std)
            return list(set(cleaned))

        l2_raw = data.get('2', [])
        l2_vars = clean_name(l2_raw[0]) if l2_raw else []

        structure = {
            'L2': l2_vars,
            'L3': []
        }

        for group in data.get('3', []):
            structure['L3'].extend(clean_name(group))

        # Remove duplicates and the target variable.
        structure['L2'] = [v for v in dict.fromkeys(structure['L2']) if v != 'T2m']
        structure['L3'] = [v for v in dict.fromkeys(structure['L3']) if v != 'T2m']

        # Keep terrain priors in L3.
        terrain_priors = ['DEM', 'Slope', 'Aspect']
        for var in terrain_priors:
            structure['L2'] = [item for item in structure['L2'] if item != var]
            if var not in structure['L3']:
                structure['L3'].append(var)
                print(f"Added {var} to L3 terrain priors.")

        # Use soil moisture as a fallback L3 prior.
        weak_priors = ['SM']
        for var in weak_priors:
            if var not in structure['L2'] and var not in structure['L3']:
                structure['L3'].append(var)
                print(f"Added {var} to L3 auxiliary priors.")

        return structure

    def forward(self, batch_data):
        # Assemble predictor channels.
        all_keys = ['DEM', 'LST', 'NDVI', 'Albedo', 'SM', 'Wind', 'Slope', 'Aspect']

        if not batch_data:
            raise ValueError("batch_data is empty")

        ref = next(iter(batch_data.values()))
        B, _, H, W = ref.shape
        device = ref.device

        input_list = []
        for k in all_keys:
            if k in batch_data:
                input_list.append(batch_data[k])
            else:
                # Fill missing predictors with zeros.
                input_list.append(torch.zeros(B, 1, H, W, device=device))

        x_in = torch.cat(input_list, dim=1)

        # Assemble conditioning groups.
        conditions = {}
        for group in ['L2', 'L3']:
            if self.sft_config[group] > 0:
                tensors = []
                for v in self.causal_structure[group]:
                    if v in batch_data:
                        tensors.append(batch_data[v])
                    else:
                        tensors.append(torch.zeros(B, 1, H, W, device=device))

                conditions[group] = torch.cat(tensors, dim=1)

        feat = self.head(x_in)
        for block in self.body:
            feat = block(feat, conditions)
        out = self.tail(feat)

        # Predict temperature directly.
        return out
