def reshape_sample(sample):
    # How I make samples go to wide format:
    batch_size = sample.shape[0]
    wide_samples = torch.zeros(batch_size,3,256,306*6) 
    for s in range(sample.shape[0]):
        for i in range(sample.shape[1]):
            wide_samples[s,:,0:256,0+306*i:306*i + 306] =  sample[s][i]
    return wide_samples
#and basically plug this in as soon as any sample is extracted from the data loader

