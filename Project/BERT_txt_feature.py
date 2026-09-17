import os
import re
import string
import time
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
from torch import nn



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} device.')

PROJECTION_STATE_NAME = 'hcbs_text_projection.pt'


class CustomModel(nn.Module):
    def __init__(self, bert_model, output_size):
        super(CustomModel, self).__init__()
        self.bert = bert_model
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 512)
        self.fc2 = nn.Linear(512, output_size)
        self.layer_norm = nn.LayerNorm(512)
        self.activation = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        pooled_output = (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        x = self.fc1(pooled_output)
        x = self.layer_norm(x)
        x = self.activation(x)
        output = self.fc2(x)
        return output.view(-1, 64, 72, 72)


def convert_sentence_to_tensor(sentence, tokenizer):
    tokens = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
    return tokens['input_ids'], tokens['attention_mask']


def convert_dataset_to_tensors(dataset, tokenizer):
    input_tensors = []
    print('converting dataset to tensors...')
    for i in tqdm(range(len(dataset))):
        input_ids, attention_mask = convert_sentence_to_tensor(dataset[i], tokenizer)
        input_tensors.append((input_ids, attention_mask))
    return input_tensors


def train_model(*args, **kwargs):
    raise RuntimeError(
        'The former zero-target objective collapses semantic features and has been disabled. '
        'Export a verified trained BERT+projection checkpoint. A replacement training objective '
        'requires an explicitly documented research protocol and new benchmark results.')


def save_model(model, tokenizer, path):
    os.makedirs(path, exist_ok=True)
    model.bert.save_pretrained(path)
    tokenizer.save_pretrained(path)
    projection_state = {
        'fc1': model.fc1.state_dict(),
        'fc2': model.fc2.state_dict(),
        'layer_norm': model.layer_norm.state_dict(),
    }
    torch.save(projection_state, os.path.join(path, PROJECTION_STATE_NAME))


def load_model(path, output_size):
    model = CustomModel(BertModel.from_pretrained(path), output_size)
    projection_path = os.path.join(path, PROJECTION_STATE_NAME)
    if os.path.exists(projection_path):
        state = torch.load(projection_path, map_location='cpu', weights_only=True)
        model.fc1.load_state_dict(state['fc1'])
        model.fc2.load_state_dict(state['fc2'])
        model.layer_norm.load_state_dict(state['layer_norm'])
    else:
        raise FileNotFoundError(
            'Missing {} in {}. Older HCBS checkpoints saved only BERT weights and cannot '
            'reconstruct the trained projection layers.'.format(PROJECTION_STATE_NAME, path)
        )
    return model


def test_model(model, sentence, tokenizer):
    model.to(device)
    model.eval()
    with torch.no_grad():
        input_ids, attention_mask = convert_sentence_to_tensor(sentence, tokenizer)
        output = model(input_ids.to(device), attention_mask.to(device))
    return output


def get_sentences(path='content_sentence'):
    print('load data...')
    text_list = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    translator = str.maketrans('', '', string.punctuation)
                    text = text.translate(translator)
                    text = re.sub(r'\n+', ' ', text)
                    text_list.append(text.lower())
    print('load data success!')
    return text_list


def get_sentence(path):
    if not path.endswith('.txt'):
        raise ValueError('Expected a .txt file, got: {}'.format(path))
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    text = re.sub(r'\n+', ' ', text)
    return text.lower()


def get_txt_files_in_folder(folder_path):
    txt_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                txt_files.append(os.path.join(root, file))
    return txt_files


def digest(path):
    value = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def save_tensor(tensor, target):
    array = tensor.detach().cpu().numpy()
    if array.shape != (1, 64, 72, 72) or not np.isfinite(array).all():
        raise ValueError('Expected a finite [1,64,72,72] projection output')
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.npy.partial')
    with temporary.open('wb') as stream:
        np.save(stream, array[0].astype(np.float32), allow_pickle=False)
    os.replace(temporary, target)


def use_model(sentence_path, model_path, output_root):
    source = Path(sentence_path).resolve()
    output = Path(output_root).resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose an empty output directory; existing features must remain immutable')
    files = sorted(source.rglob('*.txt'))
    if not files:
        raise ValueError('No descriptions found in {}'.format(source))
    tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True)
    model = load_model(model_path, 64 * 72 * 72).to(device).eval()
    model_root = Path(model_path).resolve()
    model_revision = {str(path.relative_to(model_root)): digest(path)
                      for path in sorted(model_root.rglob('*')) if path.is_file()}
    records = []
    output.mkdir(parents=True, exist_ok=True)
    for file in tqdm(files):
        target = output / file.relative_to(source).with_suffix('.npy')
        result = test_model(model, get_sentence(str(file)), tokenizer)
        save_tensor(result, target)
        records.append({'path': str(target.relative_to(output)), 'sha256': digest(target),
                        'source_sha256': digest(file)})
    manifest = {'schema': 1, 'format': 'feature_map', 'shape': [64, 72, 72],
                'model_files': model_revision, 'files': records,
                'note': 'Export only. No text model training or benchmark verification performed.'}
    temporary = output / 'features.json.partial'
    temporary.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    os.replace(temporary, output / 'features.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export an existing BERT+projection checkpoint; never train on test descriptions')
    parser.add_argument('--sentence_root', default=os.environ.get('HCBS_SENTENCE_PATH'))
    parser.add_argument('--model_path', default=os.environ.get('HCBS_TEXT_MODEL_PATH'))
    parser.add_argument('--output_root', required=True)
    args = parser.parse_args()
    if not args.sentence_root or not args.model_path:
        parser.error('--sentence_root and --model_path (or their HCBS environment variables) are required')
    use_model(args.sentence_root, args.model_path, args.output_root)
