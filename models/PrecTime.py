import torch
import torch.nn as nn
import torch.nn.functional as F


def conv1d_block(
    in_channels,
    out_channels,
    kernel_size=5,
    stride=1,
    padding=2,
    dilation=1,
    maxpool=False,
    dropout=False
):
    layers = [nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation
    )]
    if maxpool:
        layers.append(nn.MaxPool1d(kernel_size=2))
    if dropout:
        layers.append(nn.Dropout(p=0.5))
    return nn.Sequential(*layers)


class PrecTime(nn.Module):
    def __init__(
        self,
        input_channels : int = 2,
        use_time_cat : bool = True,
        hidden_channels : int = 128,
        kernel_size : int = 5,
        padding : int = 2,
        stride : int = 1,
        dilation : int = 1,
        sequence_length : int = 1024,
        num_classes : int = 3,
        chunks : int = 1,
        fe1_layers : int = 4,
        fe2_layers : int = 4
    ):
        super(PrecTime, self).__init__()
        
        if use_time_cat:
            # add categorical embeddings
            self.hour_encoder = torch.nn.Embedding(24, 4)
            self.day_encoder = torch.nn.Embedding(7, 4)
            input_channels += 8

        self.input_channels = input_channels
        self.use_time_cat = use_time_cat
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.chunks = chunks
        self.fe1_layers = fe1_layers
        self.fe2_layers = fe2_layers
        
        feature_extraction1_layer = []
        feature_extraction1_layer.extend([
            conv1d_block(
                in_channels=self.input_channels,
                out_channels=self.hidden_channels,
                kernel_size=self.kernel_size,
                padding=self.padding,
                stride=self.stride,
                dilation=self.dilation
            ),
            conv1d_block(
                in_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                kernel_size=self.kernel_size,
                padding=self.padding,
                stride=self.stride,
                dilation=self.dilation,
                maxpool=True,
                dropout=True
            )
        ])
        for i in range(self.fe1_layers):
            feature_extraction1_layer.extend([
                conv1d_block(
                    in_channels=self.hidden_channels,
                    out_channels=self.hidden_channels,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    stride=self.stride,
                    dilation=self.dilation
                )
            ])
        self.feature_extraction1 = nn.Sequential(
            *feature_extraction1_layer
        )
        
        feature_extraction2_layer = []
        feature_extraction2_layer.extend([
            conv1d_block(
                in_channels=self.input_channels,
                out_channels=self.hidden_channels,
                kernel_size=self.kernel_size,
                padding=8,
                stride=self.stride,
                dilation=4
            ),
            conv1d_block(
                in_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                kernel_size=self.kernel_size,
                padding=8,
                stride=self.stride,
                dilation=4,
                maxpool=True,
                dropout=True
            )
        ])
        for i in range(self.fe2_layers):
            feature_extraction2_layer.extend([
                conv1d_block(
                    in_channels=self.hidden_channels,
                    out_channels=self.hidden_channels,
                    kernel_size=self.kernel_size,
                    padding=8,
                    stride=self.stride,
                    dilation=4
                )
            ])
        
        self.feature_extraction2 = nn.Sequential(
            *feature_extraction2_layer
        )

        self.fc1 = nn.Linear(
            self.hidden_channels * 2 *
            (self.sequence_length // self.chunks // 2), 64
        )

        self.context_detection1 = nn.LSTM(
            input_size=64,
            hidden_size=100,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.context_detection2 = nn.LSTM(
            input_size=200,
            hidden_size=128,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.inter_upsample = nn.Upsample(
            scale_factor=self.sequence_length // self.chunks,
            mode='nearest'
        )
        self.inter_fc = nn.Linear(
            in_features=self.context_detection2.hidden_size * 2,
            out_features=3
        )

        self.inter_upsample_di = nn.Upsample(
            scale_factor=self.sequence_length // self.chunks // 2,
            mode='nearest'
        )
        
        self.prediction_refinement = nn.Sequential(
            conv1d_block(
                in_channels=self.hidden_channels * 2 + self.context_detection2.hidden_size * 2,
                out_channels=self.hidden_channels,
                kernel_size=self.kernel_size,
                padding=2,
                stride=self.stride,
                dilation=self.dilation,
                maxpool=False,
                dropout=False
            ),
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv1d_block(
                in_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                kernel_size=self.kernel_size,
                padding=2,
                stride=self.stride,
                dilation=self.dilation,
                maxpool=False,
                dropout=True
            ),
            nn.Dropout(p=0.5)
        )

        self.fc_final = nn.Linear(self.hidden_channels, num_classes)

    def forward(self, x):
        if self.use_time_cat:
            #use categorical embeddings
            x = torch.concat([x[...,:-2],self.hour_encoder(x[...,-2].int()),self.day_encoder(x[...,-1].int())],dim=-1)
        
        origin_x = x

        if x.shape[1] % self.chunks != 0:
            print(ValueError("Seq Length Should be Divided by Num_Chunks"))

        if x.shape[-1] != self.input_channels:
            print(ValueError(
                "The Channel of Your Input should equal to Defined Input Channel"))

        if x.shape[1] != self.sequence_length:
            print(ValueError(
                "The Length of Your Input should equal to Defined Seq Length"))

        x = x.permute(0, 2, 1)
        x = x.reshape(
            -1,
            self.input_channels,
            x.shape[-1] // self.chunks
        )
        
        features1 = self.feature_extraction1(x)
        features2 = self.feature_extraction2(x)
        features_combined = torch.cat((features1, features2), dim=1)

        features_combined_flat = features_combined.view(origin_x.shape[0], self.chunks, -1)
        features_combined_flat = self.fc1(features_combined_flat)
        
        context1, _ = self.context_detection1(features_combined_flat)
        context2, _ = self.context_detection2(context1)
        
        output1 = context2.permute(0, 2, 1)
        output1 = self.inter_upsample(output1)
        output1 = output1.permute(0, 2, 1)
        output1 = self.inter_fc(output1)
        
        di = context2.permute(0, 2, 1)
        di = self.inter_upsample_di(di)
        ui = features_combined.transpose(0, 1).reshape(
            features_combined.shape[1], origin_x.shape[0], -1
        ).transpose(0, 1)
        combine_ui_di = torch.cat([ui, di], dim=1)
        
        final_output = self.prediction_refinement(combine_ui_di)
        final_output = self.fc_final(final_output.permute(0, 2, 1))
        
        return final_output