import argparse

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from models.vgg import vgg11_bn
from models.self_attn import SelfAttention
from tb_writer import TensorboardWriter
import os
from dataloader import get_loader


parser = argparse.ArgumentParser(description='Depth Completion')
parser.add_argument('-b', '--batch_size', type=int, default=16, help='batch size')
parser.add_argument('-e', '--epoch', type=int, default=1000, help='number of epochs')
parser.add_argument('-m', '--saved_model_name', type=str, default='model', help='saved_model_name')
parser.add_argument('-cpu', '--using_cpu', action='store_true', help='use cpu')
parser.add_argument('-l', '--load_model', help='load model')
parser.add_argument('-n', '--num_data', type=int, default=20000, help='the number of data used to train')
parser.add_argument('-f', '--finetune', action='store_true', help='finetune encoder or not')


args = parser.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() and not args.using_cpu else 'cpu'

def train_val(encoder, decoder, criterion, optimizer, loader, epoch, device):
    train_acc, val_acc = 0, 0
    train_loss, val_loss = 0, 0
    for phase in ['train', 'val']:
        total_loss = 0
        total_correct = 0
        total_data = 0

        data_loader = loader[phase]
        pbar = tqdm(iter(data_loader))

        if phase == 'train':
            encoder.train()
            decoder.train()
        else:
            encoder.eval()
            decoder.eval()

        for num_batch, (seq, labels) in enumerate(pbar):
            seq = seq.to(device) # b x 50 x 2 x 256 x 256
            labels = labels.to(device)

            if phase == 'train':
                features = encoder(seq[0]).unsqueeze(1) # (50, 1, 512, 8, 8) gibi olmalı
                L, B, X, Y, Z = features.size() # Tekrar unpack etmeyi deneyeceğiz
                features = features.view(L, B, -1) # (50, 1, 32768) gibi olacak
                preds = decoder(features)
            else:
                with torch.no_grad():
                    features = encoder(seq[0]).unsqueeze(1)
                    L, B, X, Y, Z = features.size()
                    features = features.view(L, B, -1)
                    preds = decoder(features)

            pred_class = (preds > 0.5).int()
            num_correct = torch.sum((pred_class == labels)).item()

            loss = criterion(preds, labels)        

            total_correct += num_correct
            total_loss += loss.item()
            total_data += seq.size(0)

            if phase == 'train':
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_acc = total_correct/total_data*100
                train_loss = total_loss/total_data
            else:
                val_acc = total_correct/total_data*100
                val_loss = total_loss/total_data

            pbar.set_description('[{}] Epoch: {}; loss: {:.4f}, acc: {:.2f}%'.format(phase.upper(), epoch, total_loss/total_data,
                            total_correct/total_data*100))
    return train_loss, val_loss, train_acc, val_acc

class EarlyStop():
    """Early stop training if validation loss didn't improve for a long time"""
    def __init__(self, patience, mode = 'min'):
        self.patience = patience
        self.mode = mode

        self.best = float('inf') if mode == 'min' else 0
        self.cur_patience = 0

    def stop(self, loss, encoder, decoder, epoch, saved_model_path):
        update_best = loss < self.best if self.mode == 'min' else loss > self.best

        if update_best:
            self.best = loss
            self.cur_patience = 0

            torch.save({'val_loss': loss, \
                        'state_dict': decoder.state_dict(), \
                        'epoch': epoch}, saved_model_path+'.tar')
            torch.save({'val_loss': loss, \
                'state_dict': encoder.state_dict(), \
                'epoch': epoch}, saved_model_path+'_encoder.tar')
            print('SAVE MODEL to {}'.format(saved_model_path))
        else:
            self.cur_patience += 1
            if self.patience == self.cur_patience:
                return True
        
        return False


def main():
    # Initialize tensorboard
    tensorboard_path = 'runs/{}'.format(args.saved_model_name)
    tb_writer = TensorboardWriter(tensorboard_path)

    # Initialze early stop
    early_stop = EarlyStop(patience=5, mode='max')

    # Get dataloader
    train_loader = get_loader('train', batch_size=1, shuffle=True, num_data=-1)
    val_loader = get_loader('val', batch_size=1, shuffle=False, num_data=-1)
    loader = dict(train=train_loader, val=val_loader)

    # --- ENCODER OLUŞTURMA BÖLÜMÜ ---
    # Adım 1: Tam VGG11 modelini oluşturun. 
    # num_classes veya diğer parametrelerin burada ne olduğu önemli değil, çünkü son katmanlar zaten kaldırılacak.
    encoder = vgg11_bn(pretrained=False, num_classes=1, in_channels=2).to(DEVICE)
    
    # Adım 2: Eğer bir ön-eğitimli model yüklenmesi isteniyorsa, state_dict'i bu tam modele yükleyin.
    if args.load_model: # loader pretrained encoder
        print(f"Loading pretrained encoder from: {args.load_model}")
        dic = torch.load(args.load_model, map_location=DEVICE) # Cihaz uyumluluğu için map_location eklemek iyi bir pratiktir.
        state_dict = dic["state_dict"]
        encoder.load_state_dict(state_dict)
        
    # Adım 3: Model yüklensin ya da yüklenmesin, HER ZAMAN son katmanları kesin.
    # VGG'nin yapısı genellikle [features, avgpool, classifier] şeklindedir.
    # [:-2] ile 'avgpool' ve 'classifier' katmanlarını kaldırıyoruz.
    # Böylece encoder her zaman sadece konvolüsyonel özellik çıkarıcı (feature extractor) olur.
    modules = list(encoder.children())[:-2]
    encoder = nn.Sequential(*modules)
    # --- ENCODER OLUŞTURMA BÖLÜMÜ SONU ---
    
    # Decoder'ı ve diğer bileşenleri oluşturun
    #decoder = SeqModel().to(DEVICE)
    decoder = SelfAttention().to(DEVICE)
    criterion = nn.BCELoss(reduction='mean')
    
    # Optimizer'ı yapılandırın
    if args.finetune:
        print("Finetuning mode enabled. Training both encoder and decoder.")
        optimizer = optim.Adam([
            {'params': encoder.parameters(), 'lr': 1e-5}, 
            {'params': decoder.parameters(), 'lr': 1e-4}
        ])
    else:
        print("Feature extraction mode. Training only the decoder.")
        # Eğer finetune değilse, encoder'ın ağırlıklarını dondurmak iyi bir pratiktir.
        for param in encoder.parameters():
            param.requires_grad = False
        optimizer = optim.Adam(decoder.parameters(), lr=1e-4)

    # Eğitim döngüsünü başlatın
    for epoch in range(1, 100):
        # train_val fonksiyonu artık orijinal haliyle doğru bir şekilde çalışacaktır.
        train_loss, val_loss, train_acc, val_acc = train_val(encoder, decoder, criterion, optimizer, loader, epoch, DEVICE)
        
        # Tensorboard'a yaz
        tb_writer.tensorboard_write(epoch, train_loss, val_loss, train_acc, val_acc)

        # Modeli kaydet ve erken durdurmayı kontrol et
        saved_model_path = os.path.join("trained_model", "{}".format(args.saved_model_name))
        if early_stop.stop(val_acc, encoder, decoder, epoch, saved_model_path):
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    tb_writer.close()

if __name__ == '__main__':
    main()


# TODO
# inference kodu yazılacak - DONE
# seq veriyi nasıl encoder alıyor. alamaması lazım mantıken
# human watchable flow videosu oluşturma olacak. - DONE