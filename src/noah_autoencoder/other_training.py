import os 
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision
from noah_autoencoder import *

from src.utils.data_helper import UnlabeledDataset

from src.utils.helper import collate_fn, draw_box

## Neccessary imports

batch_size   = 4
epochs       = 25 #30 before, 0 to go immediately to test... 


# optimizer parameter
learning_rate = 1e-3
weight_decay  = 0


model = Car_Autoencoder(Inception_Autoencoder,Encoder,Decoder,InceptionE,BasicConv2d,in_ch=3,base_ch=32)

f_out = 'autoencoder_results_%.1e_%.1e_%d_%s.txt'\
               %(learning_rate, weight_decay, batch_size,str(epochs))

# best-model fname
f_best_model = 'BestAutoencoder_%.1e_%.1e_%d.pt'\
               %(learning_rate, weight_decay, batch_size)

if os.path.exists(f_best_model):  
    print('loading best model...')
    model.load_state_dict(torch.load(f_best_model))

total_params = sum(p.numel() for p in model.parameters())
print('Total number of parameters in network: %d'%total_params)


transform = torchvision.transforms.ToTensor()

# Not importing labeled dataset for now. 


# All the images are saved in image_folder
# All the labels are saved in the annotation_csv file
image_folder = '/Users/noahkasmanoff/Desktop/Deep_Learning/car/dat/data/'
#image_folder = '/scratch/nsk367/pytorch-use/DLSP20/dat/data/ '#'../dat/data/'
annotation_csv = image_folder + 'annotation.csv'

unlabeled_scene_index_train = np.arange(1)
unlabeled_scene_index_valid = np.arange(1,2)

unlabeled_trainset = UnlabeledDataset(image_folder=image_folder, scene_index=unlabeled_scene_index_train, 
                                      first_dim='sample', transform=transform)
unlabeled_trainloader = torch.utils.data.DataLoader(unlabeled_trainset, batch_size=batch_size, 
                                                    shuffle=True, num_workers=2)

unlabeled_validset = UnlabeledDataset(image_folder=image_folder, scene_index=unlabeled_scene_index_valid, 
                                      first_dim='sample', transform=transform)
unlabeled_validloader = torch.utils.data.DataLoader(unlabeled_validset, batch_size=batch_size, 
                                                    shuffle=True, num_workers=2)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
print("device " , device)
criterion = nn.SmoothL1Loss() #nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5,   patience=5, verbose=True)




# load best-model in case it exists
if os.path.exists(f_best_model):  
    print('loading best models...')
    
    model.load_state_dict(torch.load(f_best_model))


model.eval()

count, best_loss = 0, 0.0
with torch.no_grad():
    for sample in unlabeled_validloader:
        #send objects to device
        sample = sample.to(device)
        #feed forward classifier,compute loss    
        x_pred, z = model(sample)
        target = F.pad(sample[:,3],(-2,-1,-1,0))
        error    = criterion(x_pred,target)
        best_loss += error.cpu().numpy()
        
        
        count += 1

best_loss /= count
print('validation error = %.3e'%best_loss)


# Main loop. First, remove out files in case of overlap. 
if os.path.exists(f_out):  os.system('rm %s'%f_out)


    
for epoch in range(epochs):
    # TRAIN
    model.train()
    count, loss_train = 0, 0.0
    for sample in unlabeled_trainloader:  
        #send to device
        sample = sample.to(device)

        # Forward Pass
        optimizer.zero_grad()
        x_pred , z = model(sample)
        target = F.pad(sample[:,3],(-2,-1,-1,0))
        loss = criterion(x_pred,target)

        loss_train += loss.cpu().detach().numpy()


        # Backward Pass
        loss.backward()
        optimizer.step()   
        count += 1
    loss_train /= count

    # VALID
    model.eval()
    count, loss_valid = 0,0.0
    with torch.no_grad():
        for sample in unlabeled_validloader:
            #send objects to device
            sample = sample.to(device)
            #feed forward classifier,compute loss    
            x_pred, z = model(sample)
            target = F.pad(sample[:,3],(-2,-1,-1,0))
            error    = criterion(x_pred,target)
            loss_valid += error.cpu().numpy()
 
            count += 1
    
    # Save Best Models 
    if loss_valid<best_loss:
        best_loss = loss_valid
        torch.save(model.state_dict(), f_best_model)
        print('%03d %.4e %.4e (saving)'\
              %(epoch, loss_train, loss_valid))

    else:
        print('%03d %.4e %.4e'%(epoch, loss_train, loss_valid))


        
    # update learning rate
    scheduler.step(loss_valid) #also, return to previous best? 
    # save results to file
    f = open(f_out, 'a')
    f.write('%d %.4e %.4e \n'%(epoch, loss_train, loss_valid))
    f.close()

        