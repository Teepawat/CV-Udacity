import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first = True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.hidden = (torch.zeros(1, 1, self.hidden_size))
    
    def forward(self, features, captions):
        embeddings = self.embed(captions[:,:-1])
        features = features.view(len(features), 1, -1)
        inputs = torch.cat((features, embeddings), 1)
        out, hidden = self.lstm(inputs)
        output = self.linear(out)
        return output
        
    def sample(self, inputs, states=None, max_len=20):
        outputs_ids = []
        for i in range(max_len):
            hiddens, states = self.lstm(inputs.view(1,1,-1), states)
            outputs = self.linear(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            outputs_ids.append(predicted)
            inputs = self.embed(predicted)          
        return [int(i.cpu().numpy()) for i in outputs_ids]
        
        
#     def sample(self, inputs, states = None):
#         " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
#         # maximum sampling length
#         hiddens, states = self.lstm(inputs, states)        # (batch_size, 1, hidden_size), 
#         outputs = self.linear(hiddens.squeeze(1))          # (batch_size, vocab_size)
#         #outputs = outputs.max(1)
#         #sampled_ids.append(predicted)
#         #inputs = self.embed(predicted)
#         #inputs = inputs.unsqueeze(1)                       # (batch_size, 1, embed_size)
#         #sampled_ids = torch.cat(sampled_ids, 1)                # (batch_size, 20)
#         return outputs #outputs.int().squeeze().tolist() #sampled_ids.squeeze()