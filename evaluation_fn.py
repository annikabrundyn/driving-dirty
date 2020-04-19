def threat_score(prediction, target):
    """
    Inputs:
    prediction: (batch_size, 800,800) tensor with entries in {0,1}
    target: (batch_size,800,800) tensor with entries in {0,1}

    Calculates TP, FP, FN

    Output:
    Average threat score
    """

    n = torch.stack(target).shape[0]

    # Initialize
    sum_ts = 0

    for i in range(n):
        # prediction * target = true positive
        tp_tensor = torch.mul(prediction[i],target[i])

        # prediction - true positive = false positive
        fp_tensor = torch.add(prediction[i], tp_tensor[i], alpha = -1)

        # target - true positive = false negative
        fn_tensor = torch.add(target[i], tp_tensor[i], alpha = -1)

        tp = np.int(torch.sum(tp_tensor))
        fp = np.int(torch.sum(fp_tensor))
        fn = np.int(torch.sum(fn_tensor))

        # threat score
        ts = tp/(tp + fp + fn)

        sum_ts += ts

    sum_ts /= n

    return sum_ts
