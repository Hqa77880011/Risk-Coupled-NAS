import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms


class Network(nn.Module):
    def __init__(self, genotype, classes):
        super().__init__()
        depth, width, kernel, expansion, _ = genotype
        blocks = []
        channels = 3
        for layer in range(depth):
            hidden = width * expansion
            blocks.extend([nn.Conv2d(channels, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU(),
                           nn.Conv2d(hidden, hidden, kernel, padding=kernel // 2, stride=2 if layer % 2 else 1, groups=hidden, bias=False),
                           nn.BatchNorm2d(hidden), nn.ReLU(), nn.Conv2d(hidden, width, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU()])
            channels = width
        self.features = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.classifier = nn.Linear(width, classes)

    def forward(self, x):
        return self.classifier(self.features(x))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['train', 'measure', 'test'])
    p.add_argument('--genotype', default='[4,32,3,2,32]')
    p.add_argument('--dataset', choices=['cifar10', 'cifar100'], default='cifar10')
    p.add_argument('--architecture', default='a0')
    p.add_argument('--data', default='data')
    p.add_argument('--output', default='runs/architecture')
    p.add_argument('--device', default='cpu')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--seed', type=int, default=13)
    p.add_argument('--block', type=int, default=0)
    p.add_argument('--cap-ms', type=float, default=100)
    p.add_argument('--budget-ms', type=float, default=20)
    p.add_argument('--invocations', type=int, default=100)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    genotype = json.loads(args.genotype)
    if len(genotype) != 5 or genotype[0] < 1 or genotype[1] < 1 or genotype[2] not in [3, 5, 7] or genotype[3] < 1:
        p.error('Genotype is [depth,width,odd kernel,expansion,precision bits]')
    precision = genotype[-1]
    if precision not in [8, 16, 32] or (precision == 16 and not args.device.startswith('cuda')):
        p.error('Supported deployment precision is CPU INT8/FP32 or CUDA FP16/FP32')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    classes = 10 if args.dataset == 'cifar10' else 100
    cls = CIFAR10 if classes == 10 else CIFAR100
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([.5] * 3, [.5] * 3)])
    training = cls(args.data, train=True, download=True, transform=transform)
    order = np.random.default_rng(args.seed).permutation(len(training))
    boundary = int(.8 * len(order))
    train_data = torch.utils.data.Subset(training, order[:boundary])
    development = torch.utils.data.Subset(training, order[boundary:])
    model = Network(genotype, classes).to(args.device)
    if args.command == 'train':
        optimizer = torch.optim.SGD(model.parameters(), lr=.05, momentum=.9, weight_decay=.0005)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
        for _ in range(args.epochs):
            model.train()
            for images, labels in torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True):
                loss = F.cross_entropy(model(images.to(args.device)), labels.to(args.device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
        model.eval()
        correct, count = 0, 0
        with torch.no_grad():
            for images, labels in torch.utils.data.DataLoader(development, batch_size=128):
                correct += int((model(images.to(args.device)).argmax(1).cpu() == labels).sum())
                count += len(labels)
        torch.save(model.cpu().state_dict(), out / 'weights.pt')
        (out / 'development.json').write_text(json.dumps(dict(error=1 - correct / count, genotype=genotype, dataset=args.dataset)))
        return
    model.load_state_dict(torch.load(out / 'weights.pt', map_location=args.device, weights_only=True))
    model.eval()
    if precision == 8:
        if args.device != 'cpu':
            p.error('INT8 path uses the x86 CPU quantization backend')
        from torch.ao.quantization import get_default_qconfig_mapping
        from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
        model = prepare_fx(model, get_default_qconfig_mapping('x86'), (torch.zeros(1, 3, 32, 32),))
        with torch.no_grad():
            for images, _ in torch.utils.data.DataLoader(train_data, batch_size=128):
                model(images)
                break
        model = convert_fx(model)
    elif precision == 16:
        model = model.half()
    if args.command == 'test':
        test = cls(args.data, train=False, download=True, transform=transform)
        correct, count = 0, 0
        with torch.no_grad():
            for images, labels in torch.utils.data.DataLoader(test, batch_size=128):
                images = images.to(args.device)
                images = images.half() if precision == 16 else images
                correct += int((model(images).argmax(1).cpu() == labels).sum())
                count += len(labels)
        (out / 'test.json').write_text(json.dumps(dict(accuracy=correct / count, examples=count)))
        return
    example = torch.zeros(1, 3, 32, 32, device=args.device)
    example = example.half() if precision == 16 else example
    with torch.no_grad():
        for _ in range(20):
            model(example)
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.invocations):
            model(example)
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        cost = (time.perf_counter() - start) * 1000 / args.invocations
    correct, count = 0, 0
    with torch.no_grad():
        for images, labels in torch.utils.data.DataLoader(development, batch_size=128):
            images = images.to(args.device)
            images = images.half() if precision == 16 else images
            correct += int((model(images).argmax(1).cpu() == labels).sum())
            count += len(labels)
    error = 1 - correct / count
    record = dict(architecture=args.architecture, device=args.device, objective='latency', block=args.block,
                  cost=cost, cap=args.cap_ms, budget=args.budget_ms, charge=1, error=error,
                  split='search', genotype=json.dumps(genotype))
    path = out / 'measurement.csv'
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(record))
        writer.writeheader()
        writer.writerow(record)


if __name__ == '__main__':
    main()
