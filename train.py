from vocab import build_vocab
from dataloader import get_dataloader
from model import Encoder, Decoder, Seq2Seq
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def train(model,trainloader,epochs,optimizer,criterion,device):
    
    model.train()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        print('*'*20 + f'Epoch {epoch+1}' + '*'*20)
        for src,tgt in trainloader:
            src = src.to(device)
            tgt = tgt.to(device)
            optimizer.zero_grad()
            output = model(src,tgt)
            
            # Ignore <SOS> token in each sequence
            tgt = tgt[:,1:]
            
            # Print the 1st sequnece in output by taking the argmax  and also print the 1st sequence in target
            # print('Output:', output.argmax(2)[0])
            # print('Target:', tgt[0])
            
            # print(predict(model,'fe 29 2090',input_vocab,output_vocab,output_vocab_inv,max_output_len,device))
           
            output_dim = output.shape[-1]
            output = output.reshape(-1,output_dim)
            tgt = tgt.reshape(-1)
    
            loss = criterion(output,tgt)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch: {epoch+1:02}')
        
        
        print(f'Train Loss: {epoch_loss/len(trainloader):.3f}')
        print(f'Validation Loss: {evaluate(model,validloader,criterion,device):.3f}')
        
    # Save the model in Models folder
    torch.save(model.state_dict(), 'Models/model.pth')
    

def evaluate(model,validloader,criterion,device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src,tgt in validloader:
            src = src.to(device)
            tgt = tgt.to(device)
            output = model(src,tgt,0) #turn off teacher forcing
            
           
            
            output_dim = output.shape[-1]
            output = output.reshape(-1,output_dim)
            tgt = tgt[:,1:]
            tgt = tgt.reshape(-1)
            
            loss = criterion(output,tgt)
            epoch_loss += loss.item()
    return epoch_loss/len(validloader)

def predict(model,src,src_vocab,tgt_vocab,tgt_inv_vocab,max_len,device):
    
 
    src = torch.tensor([src_vocab.get(char,src_vocab['<UNK>']) for char in src]).unsqueeze(0).to(device)
    tgt = [tgt_vocab['<SOS>']]+[tgt_vocab['<PAD>']]*max_len+[tgt_vocab['<EOS>']]
    tgt = torch.tensor(tgt).unsqueeze(0).to(device)
    
    outputs = model(src,tgt,0)
    
    outputs = outputs.squeeze(0)
   
    decoder_outputs = []
    for output in outputs:
            output = output.argmax(0).item()
            
            if output == tgt_vocab['<EOS>']:
                break
            decoder_outputs.append(tgt_inv_vocab[output])
    return "".join(decoder_outputs)



input_vocab, output_vocab, input_vocab_inv,output_vocab_inv = build_vocab('Data/train.txt')
input_vocab_size = len(input_vocab)
output_vocab_size = len(output_vocab)
max_input_len = 20
max_output_len = 10
batch_size = 32
embedding_size = 128
enc_hidden_size = 128
dec_hidden_size = 2*128
learning_rate = 0.001
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainloader = get_dataloader('Data/train.txt', input_vocab, output_vocab, max_input_len, max_output_len, batch_size)
validloader = get_dataloader('Data/validation.txt', input_vocab, output_vocab, max_input_len, max_output_len, batch_size)


encoder = Encoder(input_vocab_size, embedding_size, enc_hidden_size)
decoder = Decoder(output_vocab_size, embedding_size, enc_hidden_size, dec_hidden_size)
model = Seq2Seq(encoder, decoder, device).to(device)

criterion = nn.CrossEntropyLoss(ignore_index = output_vocab['<PAD>'])
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train(model,trainloader,num_epochs,optimizer,criterion,device)







