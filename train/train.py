import os
import torch
import torch.nn as nn
import numpy as np

from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error


try:
    from model.MBF_HybridNet import WheatNet
    from dataset.dataset import (
        train_dataset,
        test_dataset,
        split_data_leave_one_out,
        Batch_size
    )
except ImportError:
    print("Please configure the project structure according to README.md")

# ==========================
# Training configuration
# ==========================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


from config import (
    BATCH_SIZE,
    EPOCHS,
    LR,
    WEIGHT_DECAY
)

EPOCHS = 100

ITERATIONS = 100


OUTPUT_DIR = "./results"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)



# ==========================
# Early stopping
# ==========================

class EarlyStopping:

    def __init__(
            self,
            patience=10,
            min_delta=1e-5):

        self.patience = patience
        self.min_delta = min_delta

        self.best_loss = None
        self.counter = 0
        self.stop = False


    def __call__(self,val_loss):

        if self.best_loss is None:

            self.best_loss = val_loss


        elif val_loss < self.best_loss-self.min_delta:

            self.best_loss = val_loss
            self.counter = 0


        else:

            self.counter += 1


            if self.counter >= self.patience:

                self.stop = True



# ==========================
# Evaluation
# ==========================

def evaluate(
        model,
        loader):


    model.eval()


    predictions=[]
    targets=[]


    with torch.no_grad():

        for data,target in loader:


            data=data.to(DEVICE)

            target=target.to(DEVICE)


            pred=model(data)


            predictions.extend(
                pred.cpu().numpy()
            )

            targets.extend(
                target.cpu().numpy()
            )


    predictions=np.array(predictions)

    targets=np.array(targets)


    return {

        "MAE":
        mean_absolute_error(
            targets,
            predictions
        ),

        "R2":
        r2_score(
            targets,
            predictions
        )
    }



# ==========================
# Training
# ==========================

def train(
        feature_data,
        label_data,
        auxiliary_data):


    scaler=MinMaxScaler()


    label_data=scaler.fit_transform(
        label_data
    )


    for (
        train_x,
        test_x,
        train_y,
        test_y,
        train_aux,
        test_aux,
        _
    ) in split_data_leave_one_out(
        feature_data,
        label_data,
        auxiliary_data
    ):


        model=WheatNet().to(
            DEVICE
        )


        optimizer=Adam(
            model.parameters(),
            lr= LR,
            weight_decay=WEIGHT_DECAY
        )


        criterion=nn.MSELoss()


        train_loader=DataLoader(
            train_dataset(
                train_x,
                train_y,
                train_aux
            ),
            batch_size=Batch_size,
            shuffle=True
        )


        test_loader=DataLoader(
            test_dataset(
                test_x,
                test_y,
                test_aux
            ),
            batch_size=Batch_size
        )


        early_stop=EarlyStopping()


        best_r2=-999



        for epoch in range(EPOCHS):


            model.train()


            for batch in train_loader:


                inputs,targets,aux=batch


                inputs=inputs.to(DEVICE)

                targets=targets.to(DEVICE)

                aux=aux.to(DEVICE)


                prediction=model(
                    [
                        inputs,
                        None,
                        aux
                    ]
                )


                loss=criterion(
                    prediction,
                    targets
                )


                optimizer.zero_grad()

                loss.backward()

                optimizer.step()



            metrics=evaluate(
                model,
                test_loader
            )


            print(
                f"Epoch:{epoch}",
                metrics
            )


            if metrics["R2"]>best_r2:


                best_r2=metrics["R2"]


                torch.save(
                    model.state_dict(),
                    os.path.join(
                        OUTPUT_DIR,
                        "best_model.pth"
                    )
                )


            early_stop(
                metrics["MAE"]
            )


            if early_stop.stop:

                break



if __name__=="__main__":


    print(
        "Please prepare dataset according to README."
    )