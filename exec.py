import torch
import torch.optim as optim
import torch.nn.functional as nnf
from torchvision import models
import prep

# функция стайл трансфера
def style_transfer(model, optimizer, content_img, style1_img, style2_img, output_img, num_iter=400, alpha=1, beta=0.01):
    style_weights = {'conv1_1': 0.3,
                     'conv2_1': 0.5,
                     'conv3_1': 0.3,
                     'conv4_1': 0.2,
                     'conv5_1': 0.2}

    content_features = prep.get_features(content_img, model)
    style1_features = prep.get_features(style1_img, model)
    style2_features = prep.get_features(style2_img, model)

    style1_grams = {
        layer: prep.gram_matrix(style1_features[layer]) for layer in style1_features
    }

    style2_grams = {
        layer: prep.gram_matrix(style2_features[layer]) for layer in style2_features
    }

    mask = prep.generate_mask(shape=content_img.shape[2:], num_points=30).to(device, dtype=torch.float).unsqueeze(0).unsqueeze(0)

    for i in range(1, num_iter+1):
        optimizer.zero_grad()
        target_features = prep.get_features(output_img, model)

        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)

        style1_loss = 0
        style2_loss = 0
        for layer in style_weights:
            target_feature = target_features[layer]
            mask_shape = target_feature.shape[2:]
            mask_res = nnf.interpolate(mask, size=mask_shape, mode='bilinear')
            mask_tensor = mask_res[:,0,:,:].expand_as(target_feature)
            target1_gram = prep.gram_matrix(target_feature*mask_tensor)
            target2_gram = prep.gram_matrix(target_feature*(mask_tensor*(-1)+1))
            _, channels, hgt, wdth = target_feature.shape
            style1_gram = style1_grams[layer]
            style2_gram = style2_grams[layer]
            layer_style1_loss = style_weights[layer] * torch.mean((target1_gram - style1_gram) ** 2)
            layer_style2_loss = style_weights[layer] * torch.mean((target2_gram - style2_gram) ** 2)

            style1_loss += layer_style1_loss / (channels * hgt * wdth)
            style2_loss += layer_style2_loss / (channels * hgt * wdth)


        total_loss = alpha * content_loss + beta * (style1_loss + style2_loss)
        total_loss.backward(retain_graph=True)
        optimizer.step()

        if i % 50 == 0:
            total_loss_rounded = round(total_loss.item(), 2)
            content_fraction = round(alpha * content_loss.item() / total_loss.item(), 2)
            style1_fraction = round(beta * style1_loss.item() / total_loss.item(), 2)
            style2_fraction = round(beta * style2_loss.item() / total_loss.item(), 2)
            print('Iteration {}, Total loss: {} - (content: {}, style1: {}, style2: {})'.format(
                i, total_loss_rounded, content_fraction, style1_fraction, style2_fraction))

    final_img = prep.tensor_to_img(output)

    return final_img, mask


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.vgg19(pretrained=True).to(device).eval()

for param in model.parameters():
    param.requires_grad_(False)

content = prep.load_img('contents/content4.jpg').to(device, dtype=torch.float)
style1 = prep.load_img('styles/style3.jpg').to(device, dtype=torch.float)
style2 = prep.load_img('styles/style4.jpg').to(device, dtype=torch.float)

output = content.clone().to(device).detach().requires_grad_(True)

adam = optim.Adam([output], lr=0.1)

result, mask = style_transfer(model, adam, content, style1, style2, output, num_iter=400, alpha=1, beta=0.1)
result.save('results/result4.jpg')