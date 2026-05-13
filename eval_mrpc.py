import torch
from datasets import load_dataset
from transformers import BertTokenizerFast, BertForSequenceClassification
from torch.utils.data import DataLoader

model_path = "./bert-for-mrpc"
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(model_path, return_dict=True)
model.eval()

dataset = load_dataset('glue', 'mrpc', split='validation')

def tokenize(examples):
    return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding="max_length", max_length=100)

toked = dataset.map(tokenize, batched=True)
toked = toked.map(lambda x: {'labels': x['label']}, batched=True)

# 在 set_format 之前保存原始文本和标签
sentences1 = toked['sentence1']
sentences2 = toked['sentence2']
labels = toked['label']

# 只保留模型需要的列，转为 torch tensor
toked.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

loader = DataLoader(toked, batch_size=16)

correct_samples = []
wrong_samples = []

with torch.no_grad():
    for batch_idx, batch in enumerate(loader):
        outputs = model(**batch)
        preds = outputs.logits.argmax(dim=-1).cpu().numpy()
        batch_labels = batch['labels'].cpu().numpy()
        start_idx = batch_idx * 16
        for i in range(len(preds)):
            idx = start_idx + i
            if idx >= len(sentences1):
                break
            sample = {
                'sentence1': sentences1[idx],
                'sentence2': sentences2[idx],
                'label': int(labels[idx]),
                'pred': int(preds[i])
            }
            if sample['label'] == sample['pred']:
                correct_samples.append(sample)
            else:
                wrong_samples.append(sample)

print("="*80)
print("MRPC 预测正确的例子（1=语义等价，0=不等价）")
print("="*80)
for i, s in enumerate(correct_samples[:10]):
    print(f"\n例{i+1}:")
    print(f"句子1: {s['sentence1']}")
    print(f"句子2: {s['sentence2']}")
    print(f"真实标签: {s['label']}  {'(等价)' if s['label']==1 else '(不等价)'}")
    print(f"预测标签: {s['pred']}  {'(等价)' if s['pred']==1 else '(不等价)'} -> 正确")

print("\n" + "="*80)
print("MRPC 预测错误的例子")
print("="*80)
for i, s in enumerate(wrong_samples[:10]):
    print(f"\n例{i+1}:")
    print(f"句子1: {s['sentence1']}")
    print(f"句子2: {s['sentence2']}")
    print(f"真实标签: {s['label']}  {'(等价)' if s['label']==1 else '(不等价)'}")
    print(f"预测标签: {s['pred']}  {'(等价)' if s['pred']==1 else '(不等价)'} -> 错误")