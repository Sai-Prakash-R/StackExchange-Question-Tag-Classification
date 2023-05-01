import torch.nn as nn


class MLPCustom(nn.Module):
    def __init__(self, embed_dim, vocab_size, hidden_dim1, hidden_dim2,
                 output_dim, non_linearity):

        super().__init__()
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.output_dim = output_dim
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.non_linearity = non_linearity

        # embedding_layer
        self.embedding = nn.EmbeddingBag(self.vocab_size, self.embed_dim)

        # hidden layer1
        self.hidden_layer1 = nn.Linear(self.embed_dim, self.hidden_dim1)

        # dropout layer 1
        self.drop1 = nn.Dropout(p=0.5)

        # batch layer norm 1
        self.batchnorm1 = nn.BatchNorm1d(num_features=self.hidden_dim1)

        # hidden layer2
        self.hidden_layer2 = nn.Linear(self.hidden_dim1, self.hidden_dim2)

        # dropout layer 2
        self.drop2 = nn.Dropout(p=0.5)

        # batch layer norm 2
        self.batchnorm2 = nn.BatchNorm1d(num_features=self.hidden_dim2)

        # output layer
        self.output_layer = nn.Linear(self.hidden_dim2, self.output_dim)

        # nonlinearity

    def forward(self, input_):
        text, offset = input_
        embed_out = self.embedding(text, offset)  # batch size, embedding_dim

        # batch size, hidden_dim1
        hout1 = self.non_linearity(self.hidden_layer1(embed_out))
        hout1 = self.batchnorm1(hout1)
        hout1 = self.drop1(hout1)

        # batch size, hidden_dim2
        hout2 = self.non_linearity(self.hidden_layer2(hout1))
        hout2 = self.batchnorm2(hout2)
        hout2 = self.drop2(hout2)

        ypred = self.output_layer(hout2)  # batch size, output_dim

        return ypred
