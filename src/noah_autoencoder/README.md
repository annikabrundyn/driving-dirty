What isn't working

Validation step wants 4 instead of 3 -- is this a versioning issue? Seemed to go away when I did 

valid_step(self,batch,batch_idx,e), where this was some dummy input e. 


The other issue was that it seemed like the x accepted from sample was wrong? 


    result = self.forward(*input, **kwargs)
  File "noah_autoencoder.py", line 118, in forward
    x,z = self.AE(x[:,3])
IndexError: index 3 is out of bounds for dimension 1 with size 3


As this error shows, it doesn't even accept the right x. So idk. 

Autoencoder accepts the entire batch; -- Batch,Cameras,Channels,H,W

What it outputs now is x[:,i], the reconstruction of the i'th image in

it also outputs z[:,i], which will just be used for the MLP and downstream task

