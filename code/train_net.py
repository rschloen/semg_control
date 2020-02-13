#!/usr/bin/env python3

from semg_network import Network, Network_enhanced


class Trainer():
    def __init__(model,optimizer,criterion,scheduler,device,epochs=10):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.max_epochs = epochs
        self.data_len = 1

        def one_epoch(phase):
            running_loss = 0.0
            cor_classify = 0.0

            for data in loader:
                if phase == 'train': self.optimizer.zero_grad()

                input, output = data
                input = torch.from_numpy(input)
                label = torch.from_numpy(np.array([label]))
                input = input.view(1,1,input.shape[0],input.shape[1])
                input = input.to(device)
                label = label.to(device)

                output = self.model(input)
                loss = self.criterion(output,label)
                running_loss += loss.item()

                if phase == 'train':
                    loss.backward()
                    self.optimizer.step()

                #Does prediction == actual class?
                cor_classify += (torch.argmax(output,dim=1) == torch.argmax(label)).sum().item()

            return running_loss, cor_classify


        def train(mode=['train','eval']):
            since = time.time()
            self.model.train()
            torch.set_grad_enabled(True)
            best_loss = float('+inf')
            best_model_wts = copy.deepcopy(self.model.state_dict())
            prev_loss = 0
            loss_cnt = 0

            print('Training...')
            for epoch in range(1,self.max_epochs+1):
                e_loss, e_classify = one_epoch(phase)
                e_acc = (e_classify.double()/len(data[phase]))*100 #percent
                print("\tEpoch: {}/{}   Loss: {:.8f}    Accuracy: {:.4f}".format(epoch,self.max_epochs,e_loss,e_acc))

                if e_loss < best_loss:
                    best_loss = e_loss
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                '''Add early stopping: if change in loss less than ... x times, stop.
                Useful check if updating properly as well'''
                if (e_loss - prev_loss) < 1e-3: loss_cnt += 1
                if loss_cnt > 2: break
                prev_loss = e_loss

        def test():
            self.model.eval()
            torch.set_grad_enabled(False)
            test_loss, test_correct = one_epoch('test')
            test_acc = (test_correct.double()/self.data_len)*100 #NEED DATA LENGTH
            print("Test Summary:")
            print("\tLoss = {:.8f}".format{test_loss})
            print("\tCorrect: {}/{}".format{test_correct,self.data_len})
            print("\tAccuracy: {:.4f}".format{test_acc})


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ## Initialize model
    # model = Network(6)
    model = Network_enhanced(6)
    # print(list(model.parameters()))

    # Initialize hyperparameters and supporting functions
    learning_rate = 0.02
    optimizer = optim.SGD(model.parameters(),lr=learning_rate)
    criterion = nn.MSELoss(reduction='mean')
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1)


if __name__ == '__main__':
    main()
