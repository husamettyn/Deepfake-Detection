from utils import get_sequential_optical_flow_no_fd
import torch
import torch.nn as nn
from models.vgg import vgg11_bn
from models.self_attn import SelfAttention
import argparse


parser = argparse.ArgumentParser(description='Deepfake Detection')
parser.add_argument('-i', '--input_path', help='the path of input video')
args = parser.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(DEVICE)

def load_model():
    print("Loading pretrained model ...")
    decoder = SelfAttention().to(DEVICE)
    dic = torch.load("saved_model/decoder.tar", map_location=DEVICE)
    state_dict = dic["state_dict"]
    decoder.load_state_dict(state_dict)

    encoder = vgg11_bn(pretrained=False, num_classes=1).to(DEVICE)
    
    dic = torch.load("saved_model/encoder.tar", map_location=DEVICE)
    state_dict = dic["state_dict"]
    encoder.load_state_dict(state_dict)
    modules = list(encoder.children())[:-2]
    encoder = nn.Sequential(*modules)

    encoder.eval()
    decoder.eval()

    return encoder, decoder

def test_seq_optical_flow(sequential_optical_flow, encoder, decoder):
    flows = torch.from_numpy(sequential_optical_flow).float().to(DEVICE)
    
    features = encoder(flows).unsqueeze(1) # L x 1 x 512 x 7 x 7
    
    Length, Batch, _, _, _ = features.size()
    features = features.view(Length, Batch, -1)
    
    preds = decoder(features)
    print(preds)

    if preds > 0.5:
        print("This is a real video")
    else:
        print("This is a fake video")

if __name__ == "__main__":
    encoder, decoder = load_model()

    sequential_optical_flow = get_sequential_optical_flow_no_fd(args.input_path)
    test_seq_optical_flow(sequential_optical_flow, encoder, decoder)

