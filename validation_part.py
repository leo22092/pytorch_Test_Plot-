
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
from inceeption_blocks import *
from pytorch_cbam import *
device="cuda"
from image_loader import *
from torchmetrics import Precision,Recall,F1Score
from torchmetrics.classification import MulticlassConfusionMatrix
import time
criterion=nn.CrossEntropyLoss()
num_classes=17
class CustomCNN(nn.Module):
    def __init__(self,in_channels=3,num_classes=17):
        super(CustomCNN,self).__init__()

        self.conv1 = Conv_Block(in_channels=in_channels,out_channels=64,kernel_size=(7,7),stride=(2,2),padding=(3,3))
        self.maxpool1=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.conv2=Conv_Block(64,192,kernel_size=3,stride=1,padding=1)
        self.maxpool2=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.inception3a=Inception_block(192,64,96,128,16,32,32)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.conv3=nn.Conv2d(256,64,kernel_size=3,stride=1,padding=1)
        self.module = cbam(64).to(device)
        self.conv4=Conv_Block(64,34,kernel_size=3,stride=1,padding=1)
        self.conv5 = Conv_Block(34, 17, kernel_size=3, stride=1, padding=1)




        self.fc1=nn.Linear(17*22*22,num_classes)

    def forward(self,x):
        x=self.conv1(x)
        x=self.maxpool1(x)
        x=self.conv2(x)
        x=self.maxpool2(x)
        x=self.inception3a(x)
        x=self.avgpool(x)
        x=self.conv3(x)
        # print("11111")
        # print(x.shape)
        x=self.module(x)
        # print("forward_1")
        x=self.conv4(x)
        # print(x.shape)
        x=self.conv5(x)
        x=x.reshape(x.shape[0],-1)
        # print(x.shape)
        x=self.fc1(x)
        # print("forward")
        return x

model_path= "/test/Model_final.pth"

model = CustomCNN().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()
x = torch.randn(32, 3, 224, 224).to(device)

# for x, y in test_loader:
#     x=x.to(device)
#
#     x = model(x)
#     _, predictions = x.max(1)
#     print(predictions)
#     print(y)
# def check_accuracy(loader,model):
#     test_loss = []
#     num_correct = 0
#     num_samples = 0
#     model.eval()
#
#     with torch.no_grad():
#         for x,y in loader :
#             x=x.to(device)
#             y=y.to(device)
#             # x=x.reshape(x.shape[0],-1)
#             scores=model(x)
#             t_loss=criterion(scores, y)
#             # test_loss[epochs]=t_loss
#
#             _,predictions=scores.max(1)
#             num_correct+=(predictions==y).sum()
#             num_samples+=predictions.size(0)
#
#
#         # confusion Matrix
#
#
#         #######################3
#         print(f"||||||||||||||Got {num_correct}/{num_samples}  with accuracy {num_correct/num_samples*100:.2f}")
#         ac=num_correct/num_samples*100
#
#         # confusion metrics
#         metric = MulticlassConfusionMatrix(num_classes=17).to(device)
#         metric(predictions, y)
#         fig_, ax_ = metric.plot()
#         fig_.savefig(f'confusion_matrix{ac}.png')  # Save as PNG image
#
#         # Precision
#         precision = Precision(task='MULTICLASS',average='macro',num_classes=num_classes).to(device)  # For overall precision across classes
#         precision(predictions, y)  # Update the metric
#         overall_precision = precision.compute()
#
#         # Recall
#         recall = Recall(task='MULTICLASS', average='macro',num_classes=num_classes).to(device)
#         recall(predictions,y)
#         overall_recall=recall.compute()
#
#         # F1 score
#         F1=F1Score(task='MULTICLASS', average='macro',num_classes=num_classes).to(device)
#         F1(predictions,y)
#         overall_F1=F1.compute()
#
#         print("Precision",overall_precision)
#         print("Recall",overall_recall)
#         print("F1",overall_F1)
#
#
#         model.train()
#         accuracy=(num_correct / num_samples) * 100
#         # if int(accuracy) > 70:
#         #     torch.save(model.state_dict(), f"Model_{accuracy}.pth")
#         #     torch.save(model, f"Model_{accuracy}.pkl")
#         #     torch.onnx.export(model, x, "CustomNet.onnx")
#         return f"{num_correct/num_samples*100:.2f}", f"{t_loss:.4f}",overall_precision.item(),overall_recall.item(),overall_F1.item()
def check_accuracy(loader, model):
    print("...................")
    test_loss = []
    num_correct = 0
    num_samples = 0
    model.eval()

    # Initialize TP, FP, FN, true_labels, pred_labels for F1 score and confusion matrix
    TP = torch.zeros(num_classes)
    FP = torch.zeros(num_classes)
    FN = torch.zeros(num_classes)
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for x, y in loader:

            x = x.to(device)
            y = y.to(device)
            start = time.time()

            scores = model(x)
            end=time.time()
            print("The time take for 1 image to pass through model is ",end-start)
            t_loss = criterion(scores, y)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

            # Update TP, FP, FN
            for i in range(num_classes):
                TP[i] += ((predictions == y) & (y == i)).sum().item()
                FP[i] += ((predictions != y) & (predictions == i)).sum().item()
                FN[i] += ((predictions != y) & (y == i)).sum().item()

            # Collect true labels and predictions for F1 score and confusion matrix
            true_labels.extend(y.cpu().numpy())
            pred_labels.extend(predictions.cpu().numpy())

    # Calculate precision and recall
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    test_accuracy=(num_correct/num_samples)*100
    # float(round(test_accuracy,2))

    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall)
    f1 = torch.mean(f1[torch.isfinite(f1)])  # Ignore NaN values

    # Calculate confusion matrix
    conf_matrix = torch.zeros(num_classes,num_classes)
    for t, p in zip(true_labels, pred_labels):
        conf_matrix[t, p] += 1
    print(f"Precision: {precision}")
    print(f"Recall: {recall}type= {type(recall)}")
    print(f"F1 Score: {f1}")
    plt.figure(figsize=(20, 16))
    sns.heatmap(conf_matrix, annot=True,fmt='.0f', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
    # print(f"Confusion Matrix:\n{conf_matrix}")
    print()
    return test_accuracy,t_loss,precision.mean(),recall.mean(),f1,conf_matrix


test_acc,test_loss,precision,recall,f1,conf_mat=check_accuracy(test_loader,model)

print(f"test_acc :{test_acc},\n test_loss:{test_loss},\n precision:{precision},\nrecall:{recall},")