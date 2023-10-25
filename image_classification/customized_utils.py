from .training import *

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex to run this example."
    )


def reset_optimizer(model_and_loss, args, optimizer_state, new_lr):
    named_parameters = list(model_and_loss.model.named_parameters())
    trainable_parameters = [(n, v) for n, v in named_parameters if v.requires_grad]
    print("length of total parameters: {}, length of trainable parameters (161): {}".
          format(len(named_parameters), len(trainable_parameters)))
    optimizer = get_optimizer(
        trainable_parameters,
        args.fp16,
        new_lr,
        args.momentum,
        args.weight_decay,
        nesterov=args.nesterov,
        bn_weight_decay=args.bn_weight_decay,
        state=optimizer_state,
        static_loss_scale=args.static_loss_scale,
        dynamic_loss_scale=args.dynamic_loss_scale,
    )
    # print("optimizer_state: {}".format(optimizer_state))

    if args.amp:
        model_and_loss, optimizer = amp.initialize(
            model_and_loss,
            optimizer,
            opt_level="O1",
            loss_scale="dynamic" if args.dynamic_loss_scale else args.static_loss_scale,
        )

    return model_and_loss, optimizer


