# # Check with fake sample
# import torch
# import numpy as np
# batch_size = 3
# samples = torch.rand(batch_size, 6, 3, 256, 306)

# Reconstruct output

def reconstruct(tensor_coords,tensor_edges):
    output = torch.tensor(
        [[tensor_coords[0]+tensor_edges[0], tensor_coords[0]+tensor_edges[0], tensor_coords[0], tensor_coords[0] ],
         [tensor_coords[1]+tensor_edges[1], tensor_coords[1], tensor_coords[1]+tensor_edges[1], tensor_coords[1]]]) 
    return output


batch = []
for i in range(samples.shape[0]):
    for i in range(50):
        object_is_present = np.random.randn(50)
        object_is_present = [1 if i<0.5 else 0 for i in object_is_present]
        object_is_present = np.sort(np.array(object_is_present))
        object_is_present = object_is_present[object_is_present==1]

        N = len(object_is_present)
        bb = []
        for i in range(N):
            coordinates = [np.random.uniform(0, 40), np.random.uniform(0, 40)]
            edges = [np.random.uniform(0, 2), np.random.uniform(0, 2)]

            coordinates = torch.tensor(coordinates)
            edges = torch.tensor((edges[0],edges[1]))

            bound_box = reconstruct(coordinates,edges)
            bb.append(bound_box)
            #print(torch.stack(tuple(bb)))
    batch.append(torch.stack(tuple(bb)))
    
batch = tuple(batch)

# # Check it works
# len(batch), batch[0].shape
