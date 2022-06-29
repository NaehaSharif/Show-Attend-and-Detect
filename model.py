#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 08:29:21 2022

@author: ecu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 17:04:06 2022

@author: ecu
"""


import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np

from efficientnet_pytorch import EfficientNet

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderCNN(nn.Module):
    """Encoder inputs images and returns feature maps"""
    def __init__(self):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

    def forward(self, images):
        features = self.resnet(images)
        # first, we need to resize the tensor to be
        # (batch, size*size, feature_maps)
        batch, feature_maps, size_1, size_2 = features.size()
        features = features.permute(0, 2, 3, 1)
        features = features.view(batch, size_1*size_2, feature_maps)

        return features
    
class EncoderEFF(nn.Module):
    """Encoder inputs images and returns feature maps"""
    def __init__(self):
        super(EncoderEFF, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b3')
        
        for param in self.model.parameters():
            param.requires_grad_(False)

    def forward(self, images):
        features = self.model.extract_features(images)
      
        # first, we need to resize the tensor to be
        # (batch, size*size, feature_maps)
        batch, feature_maps, size_1, size_2 = features.size()
        features = features.permute(0, 2, 3, 1)
        features = features.view(batch, size_1*size_2, feature_maps)

        return features
    
class DecoderGRU(nn.Module):
    """Attributes:
    - embedding_dim - specified size of embeddings;
    - hidden_dim - the size of RNN layer (number of hidden states)
    - vocab_size - size of vocabulary
    - p - dropout probability
    """
    def __init__(self, num_features, embedding_dim, hidden_dim, vocab_size,device, p =0.5):
        super(DecoderGRU, self).__init__()

        self.num_features = num_features
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        # scale the inputs to softmax
        self.sample_temp = 0.5
         # embedding layer that turns words into a vector of a specified size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # LSTM will have a single layer of size 512 (512 hidden units)
        # it will input concatinated context vector (produced by attention)
        # and corresponding hidden state of Decoder
        self.device=device
        
        self.gru =nn.GRU(num_features, hidden_dim)
        # produce the final output
        self.B1=nn.BatchNorm1d(hidden_dim)
        self.lstm = nn.LSTMCell(num_features, hidden_dim)
        # produce the final output
        self.fc = nn.Linear(hidden_dim, 256)
        self.Relu = nn.ReLU()
        self.B2=nn.BatchNorm1d(256)
        self.fc2=nn.Linear(256, vocab_size)
        # add attention layer
        self.attention = BahdanauAttention(num_features, hidden_dim)
        # dropout layer
        self.drop = nn.Dropout(p=p)
        self.drop2 = nn.Dropout(p=0.4)
        # add initialization fully-connected layers
        # initialize hidden state and cell memory using average feature vector
        # Source: https://arxiv.org/pdf/1502.03044.pdf
        self.init_h = nn.Linear(num_features, hidden_dim)
        self.sig=nn.Sigmoid()

    def forward(self, captions, features, sample_prob = 0.0):
        """Arguments
        ----------
        - captions - image captions
        - features - features returned from Encoder
        - sample_prob - use it for scheduled sampling

        Returns
        ----------
        - outputs - output logits from t steps
        - atten_weights - weights from attention network
        """
        # create embeddings for captions of size (batch, sqe_len, embed_dim)
        embed = self.embeddings(captions)
        h= self.init_hidden(features)
        seq_len = captions.size(1)
        feature_size = features.size(1)
        batch_size = features.size(0)
       
        # these tensors will store the outputs from lstm cell and attention weights
      # these tensors will store the outputs from lstm cell and attention weights
        outputs = torch.zeros(batch_size, seq_len, self.vocab_size).to(self.device)
        atten_weights = torch.zeros(batch_size, seq_len, feature_size).to(self.device)

        # scheduled sampling for training
        # we do not use it at the first timestep (<start> word)
        # but later we check if the probability is bigger than random
        for t in range(seq_len):
            sample_prob = 0.0 if t == 0 else 0.5
            use_sampling = np.random.random() < sample_prob
            if use_sampling == False:
                word_embed = embed[:,t,:]
            
            context, atten_weight = self.attention(features, h)
            # input_concat shape at time step t = (batch, embedding_dim + hidden_dim)
            input_concat = torch.unsqueeze(context, dim=0)
            #print(input_concat.size())
            h,_= self.gru(input_concat,torch.unsqueeze(h, dim=0))
            h= torch.squeeze(h)
            
            h=self.B1(h)
            h = self.drop(h)
            output = self.fc(h)
            output = self.Relu(output)
            output=self.B2(output)
            output = self.drop2(output)
            output = self.fc2(output)
            #print(output.size())
            if use_sampling == True:
                # use sampling temperature to amplify the values before applying softmax
                scaled_output = output / self.sample_temp
                scoring = F.log_softmax(scaled_output, dim=1)
                top_idx = scoring.topk(1)[1]
                word_embed = self.embeddings(top_idx).squeeze(1)
            #print(torch.squeeze(output).size())
            outputs[:, t,:] = output
            atten_weights[:, t,:] = atten_weight
        return outputs, atten_weights

    def init_hidden(self, features):

        """Initializes hidden state and cell memory using average feature vector.
        Arguments:
        ----------
        - features - features returned from Encoder

        Retruns:
        ----------
        - h0 - initial hidden state (short-term memory)
        - c0 - initial cell state (long-term memory)
        """
        mean_annotations = torch.mean(features, dim = 1)
        h0 = self.init_h(mean_annotations)
        
        return h0


    def greedy_search(self, features, max_sentence = 20):

        """Greedy search to sample top candidate from distribution.
        Arguments
        ----------
        - features - features returned from Encoder
        - max_sentence - max number of token per caption (default=20)

        Returns:
        ----------
        - sentence - list of tokens
        """

        sentence = []
        weights = []
        input_word = torch.tensor(0).unsqueeze(0).to(self.device)
        h, c = self.init_hidden(features)
        while True:
            embedded_word = self.embeddings(input_word)
            context, atten_weight = self.attention(features, h)
            # input_concat shape at time step t = (batch, embedding_dim + context size)
            input_concat = torch.cat([embedded_word, context],  dim = 1)
            h, c = self.lstm(input_concat, (h,c))
            h = self.drop(h)
            output = self.fc(h)
            scoring = F.log_softmax(output, dim=1)
            top_idx = scoring[0].topk(1)[1]
            sentence.append(top_idx.item())
            weights.append(atten_weight)
            input_word = top_idx
            if (len(sentence) >= max_sentence or top_idx == 1):
                break
        return sentence, weights


class DecoderRNN(nn.Module):
    """Attributes:
    - embedding_dim - specified size of embeddings;
    - hidden_dim - the size of RNN layer (number of hidden states)
    - vocab_size - size of vocabulary
    - p - dropout probability
    """
    def __init__(self, num_features, embedding_dim, hidden_dim, vocab_size, device,p =0.5):
        super(DecoderRNN, self).__init__()

        self.num_features = num_features
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        # scale the inputs to softmax
        self.sample_temp = 0.5
        self.device=device

        # embedding layer that turns words into a vector of a specified size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # LSTM will have a single layer of size 512 (512 hidden units)
        # it will input concatinated context vector (produced by attention)
        # and corresponding hidden state of Decoder
        
        
        # LSTM will have a single layer of size 512 (512 hidden units)
        # it will input concatinated context vector (produced by attention)
        # and corresponding hidden state of Decoder
        self.lstm = nn.LSTMCell(embedding_dim + num_features, hidden_dim)
        self.B1=nn.BatchNorm1d(hidden_dim)
      
        # produce the final output
        self.fc = nn.Linear(hidden_dim, 256)
        self.Relu = nn.ReLU()
        self.B2=nn.BatchNorm1d(256)
        self.fc2=nn.Linear(256, vocab_size)
        # dropout layer
        self.drop = nn.Dropout(p=p)
        self.drop2 = nn.Dropout(p=0.4)

	# add attention layer
        self.attention = BahdanauAttention(num_features, hidden_dim)

        # add initialization fully-connected layers
        # initialize hidden state and cell memory using average feature vector
        # Source: https://arxiv.org/pdf/1502.03044.pdf
        self.init_h = nn.Linear(num_features, hidden_dim)
        self.init_c = nn.Linear(num_features, hidden_dim)

    def forward(self, captions, features, sample_prob = 0.0):
        """Arguments
        ----------
        - captions - image captions
        - features - features returned from Encoder
        - sample_prob - use it for scheduled sampling

        Returns
        ----------
        - outputs - output logits from t steps
        - atten_weights - weights from attention network
        """
        # create embeddings for captions of size (batch, sqe_len, embed_dim)
        embed = self.embeddings(captions)
        
        h, c = self.init_hidden(features)
        seq_len = captions.size(1)
        feature_size = features.size(1)
        batch_size = features.size(0)
        # these tensors will store the outputs from lstm cell and attention weights
        outputs = torch.zeros(batch_size, seq_len, self.vocab_size).to(self.device)
        atten_weights = torch.zeros(batch_size, seq_len, feature_size).to(self.device)

        # scheduled sampling for training
        # we do not use it at the first timestep (<start> word)
        # but later we check if the probability is bigger than random
        for t in range(seq_len):
            sample_prob = 0.0 if t == 0 else 0.5
            use_sampling = np.random.random() < sample_prob
            if use_sampling == False:
                word_embed = embed[:,t,:]
            context, atten_weight = self.attention(features, h)
            # input_concat shape at time step t = (batch, embedding_dim + hidden_dim)
            input_concat = torch.cat([word_embed, context], 1)
            h, c = self.lstm(input_concat, (h,c))
            h=self.B1(h)
            h = self.drop(h)
            output = self.fc(h)
            output = self.Relu(output)
            output=self.B2(output)
            output = self.drop2(output)
            output = self.fc2(output)
            if use_sampling == True:
                # use sampling temperature to amplify the values before applying softmax
                scaled_output = output / self.sample_temp
                scoring = F.log_softmax(scaled_output, dim=1)
                top_idx = scoring.topk(1)[1]
                word_embed = self.embeddings(top_idx).squeeze(1)
            outputs[:, t, :] = output
            atten_weights[:, t, :] = atten_weight
        return outputs, atten_weights

    def init_hidden(self, features):

        """Initializes hidden state and cell memory using average feature vector.
        Arguments:
        ----------
        - features - features returned from Encoder

        Retruns:
        ----------
        - h0 - initial hidden state (short-term memory)
        - c0 - initial cell state (long-term memory)
        """
        mean_annotations = torch.mean(features, dim = 1)
        h0 = self.init_h(mean_annotations)
        c0 = self.init_c(mean_annotations)
        return h0, c0


    def greedy_search(self, features, input_word, max_sentence = 20):

        """Greedy search to sample top candidate from distribution.
        Arguments
        ----------
        - features - features returned from Encoder
        - max_sentence - max number of token per caption (default=20)

        Returns:
        ----------
        - sentence - list of tokens
        """

        sentence = []
        weights = []
        input_word = torch.tensor(0).unsqueeze(0).to(self.device)
        h, c = self.init_hidden(features)
        while True:
            embedded_word = self.embeddings(input_word)
            context, atten_weight = self.attention(features, h)
            # input_concat shape at time step t = (batch, embedding_dim + context size)
            input_concat = torch.cat([embedded_word, context],  dim = 1)
            h, c = self.lstm(input_concat, (h,c))
            h = self.drop(h)
            output = self.fc(h)
            scoring = F.log_softmax(output, dim=1)
            top_idx = scoring[0].topk(1)[1]
            sentence.append(top_idx.item())
            weights.append(atten_weight)
            input_word = top_idx
            if (len(sentence) >= max_sentence or top_idx == 1):
                break
        return sentence, weights

class DecoderRNN_S(nn.Module):
    """Attributes:
    - embedding_dim - specified size of embeddings;
    - hidden_dim - the size of RNN layer (number of hidden states)
    - vocab_size - size of vocabulary
    - p - dropout probability
    """
    def __init__(self, num_features, embedding_dim, hidden_dim, vocab_size, device,p =0.5):
        super(DecoderRNN_S, self).__init__()

        self.num_features = num_features
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        # scale the inputs to softmax
        self.sample_temp = 0.5
        self.device=device

        # embedding layer that turns words into a vector of a specified size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # LSTM will have a single layer of size 512 (512 hidden units)
        # it will input concatinated context vector (produced by attention)
        # and corresponding hidden state of Decoder
        
        
        # LSTM will have a single layer of size 512 (512 hidden units)
        # it will input concatinated context vector (produced by attention)
        # and corresponding hidden state of Decoder
        self.lstm = nn.LSTMCell(embedding_dim + num_features, hidden_dim)
        self.B1=nn.BatchNorm1d(hidden_dim)
      
        # produce the final output
        self.fc = nn.Linear(hidden_dim, vocab_size)
       
        # dropout layer
        self.drop = nn.Dropout(p=p)
       

	# add attention layer
        self.attention = BahdanauAttention(num_features, hidden_dim)

        # add initialization fully-connected layers
        # initialize hidden state and cell memory using average feature vector
        # Source: https://arxiv.org/pdf/1502.03044.pdf
        self.init_h = nn.Linear(num_features, hidden_dim)
        self.init_c = nn.Linear(num_features, hidden_dim)

    def forward(self, captions, features, sample_prob = 0.0):
        """Arguments
        ----------
        - captions - image captions
        - features - features returned from Encoder
        - sample_prob - use it for scheduled sampling

        Returns
        ----------
        - outputs - output logits from t steps
        - atten_weights - weights from attention network
        """
        # create embeddings for captions of size (batch, sqe_len, embed_dim)
        embed = self.embeddings(captions)
        
        h, c = self.init_hidden(features)
        seq_len = captions.size(1)
        feature_size = features.size(1)
        batch_size = features.size(0)
        # these tensors will store the outputs from lstm cell and attention weights
        outputs = torch.zeros(batch_size, seq_len, self.vocab_size).to(self.device)
        atten_weights = torch.zeros(batch_size, seq_len, feature_size).to(self.device)

        # scheduled sampling for training
        # we do not use it at the first timestep (<start> word)
        # but later we check if the probability is bigger than random
        for t in range(seq_len):
            sample_prob = 0.0 if t == 0 else 0.5
            use_sampling = np.random.random() < sample_prob
            if use_sampling == False:
                word_embed = embed[:,t,:]
            context, atten_weight = self.attention(features, h)
            # input_concat shape at time step t = (batch, embedding_dim + hidden_dim)
            input_concat = torch.cat([word_embed, context], 1)
            h, c = self.lstm(input_concat, (h,c))
            h=self.B1(h)
            h = self.drop(h)
            output = self.fc(h)
          
            if use_sampling == True:
                # use sampling temperature to amplify the values before applying softmax
                scaled_output = output / self.sample_temp
                scoring = F.log_softmax(scaled_output, dim=1)
                top_idx = scoring.topk(1)[1]
                word_embed = self.embeddings(top_idx).squeeze(1)
            outputs[:, t, :] = output
            atten_weights[:, t, :] = atten_weight
        return outputs, atten_weights

    def init_hidden(self, features):

        """Initializes hidden state and cell memory using average feature vector.
        Arguments:
        ----------
        - features - features returned from Encoder

        Retruns:
        ----------
        - h0 - initial hidden state (short-term memory)
        - c0 - initial cell state (long-term memory)
        """
        mean_annotations = torch.mean(features, dim = 1)
        h0 = self.init_h(mean_annotations)
        c0 = self.init_c(mean_annotations)
        return h0, c0


    def greedy_search(self, features, input_word, max_sentence = 20):

        """Greedy search to sample top candidate from distribution.
        Arguments
        ----------
        - features - features returned from Encoder
        - max_sentence - max number of token per caption (default=20)

        Returns:
        ----------
        - sentence - list of tokens
        """

        sentence = []
        weights = []
        input_word = torch.tensor(0).unsqueeze(0).to(self.device)
        h, c = self.init_hidden(features)
        while True:
            embedded_word = self.embeddings(input_word)
            context, atten_weight = self.attention(features, h)
            # input_concat shape at time step t = (batch, embedding_dim + context size)
            input_concat = torch.cat([embedded_word, context],  dim = 1)
            h, c = self.lstm(input_concat, (h,c))
            h = self.drop(h)
            output = self.fc(h)
            scoring = F.log_softmax(output, dim=1)
            top_idx = scoring[0].topk(1)[1]
            sentence.append(top_idx.item())
            weights.append(atten_weight)
            input_word = top_idx
            if (len(sentence) >= max_sentence or top_idx == 1):
                break
        return sentence, weights

class BahdanauAttention(nn.Module):
    """ Class performs Additive Bahdanau Attention.
    Source: https://arxiv.org/pdf/1409.0473.pdf

    """
    def __init__(self, num_features, hidden_dim, output_dim = 1):
        super(BahdanauAttention, self).__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # fully-connected layer to learn first weight matrix Wa
        self.W_a = nn.Linear(self.num_features, self.hidden_dim)
        # fully-connected layer to learn the second weight matrix Ua
        self.U_a = nn.Linear(self.hidden_dim, self.hidden_dim)
        # fully-connected layer to produce score (output), learning weight matrix va
        self.v_a = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, features, decoder_hidden):
        """
        Arguments:
        ----------
        - features - features returned from Encoder
        - decoder_hidden - hidden state output from Decoder

        Returns:
        ---------
        - context - context vector with a size of (1,2048)
        - atten_weight - probabilities, express the feature relevance
        """
        # add additional dimension to a hidden (need for summation later)
        decoder_hidden = decoder_hidden.unsqueeze(1)
        atten_1 = self.W_a(features)
        atten_2 = self.U_a(decoder_hidden)
        # apply tangent to combined result from 2 fc layers
        atten_tan = torch.tanh(atten_1+atten_2)
        # one score corresponds to one Encoder's output
        atten_score = self.v_a(atten_tan)
        atten_weight = F.softmax(atten_score, dim = 1)
        # first, we will multiply each vector by its softmax score
        # next, we will sum up this vectors, producing the attention context vector
        context = torch.sum(atten_weight * features,
                           dim = 1)
        atten_weight = atten_weight.squeeze(dim=2)
        return context, atten_weight
