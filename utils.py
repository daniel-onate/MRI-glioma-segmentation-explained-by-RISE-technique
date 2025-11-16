def dice_coeff(outputs, masks, smooth=1e-6):

    outputs = outputs.view(-1)
    masks = masks.view(-1)

    intersection = (outputs * masks).sum()
    dice = (2. * intersection + smooth) / (outputs.sum() + masks.sum() + smooth)

    return dice.item()