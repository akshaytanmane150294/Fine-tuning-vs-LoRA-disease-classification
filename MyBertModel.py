import torch
from torch import nn
from transformers import BertModel
from transformers import AutoModel


# classifier to be attached to last part of BERT
class Classifier(torch.nn.Module):
    def __init__(self, bert_embedding, num_classes):
        super(Classifier, self).__init__()
        hidden_layer = 100
        self.fc1 = torch.nn.Linear(bert_embedding, hidden_layer)
        self.dropout1 = torch.nn.Dropout(0.2)
        self.act1 = torch.nn.ReLU(hidden_layer) # ReLU
        self.fc2 = torch.nn.Linear(hidden_layer, num_classes)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, encoded_input):
         out1 = self.dropout1(self.fc1(encoded_input))
         out2 = self.act1(out1)
         logits = self.fc2(out2)
         probabilities = self.softmax(logits)
         return probabilities


class MyBertModel(nn.Module): # pretrained BERT + our own classifier
    def __init__(self, dropout_probability=0.2, use_dropout=True):
        super(MyBertModel, self).__init__()
        self.bert_model = AutoModel.from_pretrained('bert-base-cased')
        print(self.bert_model)
        self.use_dropout = use_dropout
        hidden_size = self.bert_model.config.hidden_size
        num_classes = 22
        self.classifier = Classifier(hidden_size, num_classes)

 # Freeze some BERT layers ---------------
 #modules = [self.bert_model.embeddings, self.bert_model.encoder.layer[:5]]
 # first 5 layers will not be finetuned
 #for module in modules:
 # for param in module.parameters():
 # param.requires_grad = False

    def forward(self, input_id, mask):
        hs, pooled_output = self.bert_model( input_ids= input_id,attention_mask=mask, return_dict=False )

        # classifier for BERT model's output
        out = self.classifier(pooled_output)
        return out
