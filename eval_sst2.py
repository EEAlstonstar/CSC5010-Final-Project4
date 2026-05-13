import torch
from datasets import load_dataset
from transformers import BertTokenizerFast, BertForSequenceClassification
from torch.utils.data import DataLoader

model_path = "./bert-for-sst2"
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(model_path, return_dict=True)
model.eval()

dataset = load_dataset('glue', 'sst2', split='validation')

def tokenize(examples):
    return tokenizer(examples['sentence'], truncation=True, padding="max_length", max_length=64)

toked = dataset.map(tokenize, batched=True)
toked = toked.map(lambda x: {'labels': x['label']}, batched=True)

# 保存原始句子和标签
sentences = toked['sentence']
labels = toked['label']

toked.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

loader = DataLoader(toked, batch_size=32)

correct_samples = []
wrong_samples = []

with torch.no_grad():
    for batch_idx, batch in enumerate(loader):
        outputs = model(**batch)
        preds = outputs.logits.argmax(dim=-1).cpu().numpy()
        start_idx = batch_idx * 32
        for i in range(len(preds)):
            idx = start_idx + i
            if idx >= len(sentences):
                break
            sample = {
                'sentence': sentences[idx],
                'label': int(labels[idx]),
                'pred': int(preds[i])
            }
            if sample['label'] == sample['pred']:
                correct_samples.append(sample)
            else:
                wrong_samples.append(sample)

print("="*80)
print("SST-2 预测正确的例子（1=正面，0=负面）")
print("="*80)
for i, s in enumerate(correct_samples[:10]):
    sentiment = "正面" if s['label'] == 1 else "负面"
    print(f"\n例{i+1}: {s['sentence']}")
    print(f"真实情感: {s['label']} ({sentiment})")
    print(f"预测情感: {s['pred']} ({'正面' if s['pred']==1 else '负面'}) -> 正确")

print("\n" + "="*80)
print("SST-2 预测错误的例子")
print("="*80)
for i, s in enumerate(wrong_samples[:10]):
    true_sentiment = "正面" if s['label'] == 1 else "负面"
    pred_sentiment = "正面" if s['pred'] == 1 else "负面"
    print(f"\n例{i+1}: {s['sentence']}")
    print(f"真实情感: {s['label']} ({true_sentiment})")
    print(f"预测情感: {s['pred']} ({pred_sentiment}) -> 错误")