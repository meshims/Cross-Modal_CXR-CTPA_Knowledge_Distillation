
"Largely taken and adapted from https://github.com/lucidrains/DALLE2-pytorch"

import click
import torch
from typing import List
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from accelerate.utils import dataclasses as accelerate_dataclasses
from torch import nn
from datasets.xray_ctpa import XrayCTPADataset
from xray2ctpa_knowledge_distilattion_dalle2 import xray2ctpa_knowledge_distillation_dalle2_trainer as local_trainer
from xray2ctpa_knowledge_distilattion_dalle2 import xray2ctpa_knowledge_distillation_dalle2_train_configs as local_train_configs
from xray2ctpa_knowledge_distilattion_dalle2 import xray2ctpa_knowledge_distillation_dalle2_trackers as tracker_local
from dalle2_pytorch.utils import Timer



cos = nn.CosineSimilarity(dim=1, eps=1e-6)

def exists(val):
    return val is not None


def all_between(values: list, lower_bound, upper_bound):
    for value in values:
        if value < lower_bound or value > upper_bound:
            return False

    return True


def make_model(
    prior_config: local_train_configs.DiffusionPriorConfig,
    train_config: local_train_configs.DiffusionPriorTrainConfig,
    device: str = None,
    accelerator: Accelerator = None,
):
    # create model from config
    diffusion_prior = prior_config.create()


    # instantiate the trainer
    trainer = local_trainer.DiffusionPriorTrainer(
        diffusion_prior=diffusion_prior,
        lr=train_config.lr,
        wd=train_config.wd,
        max_grad_norm=train_config.max_grad_norm,
        amp=train_config.amp,
        use_ema=train_config.use_ema,
        device=device,
        accelerator=accelerator,
        warmup_steps=train_config.warmup_steps,
    )

    return trainer


def create_tracker(
    accelerator: Accelerator,
    config: local_train_configs.TrainDiffusionPriorConfig,
    config_path: str,
    dummy: bool = False,
) -> tracker_local.Tracker:
    tracker_config = config.tracker

    accelerator_config = {
        "Distributed": accelerator.distributed_type
        != accelerate_dataclasses.DistributedType.NO,
        "DistributedType": accelerator.distributed_type,
        "NumProcesses": accelerator.num_processes,
        "MixedPrecision": accelerator.mixed_precision,
    }

    tracker: tracker_local.Tracker = tracker_config.create(
        config, accelerator_config, dummy_mode=dummy
    )

    tracker.save_config(config_path, config_name="prior_config.json")

    return tracker


def pad_gather_reduce(trainer: local_trainer.DiffusionPriorTrainer, x, method="mean"):
    """
    pad a value or tensor across all processes and gather

    params:
        - trainer: a trainer that carries an accelerator object
        - x: a number or torch tensor to reduce
        - method: "mean", "sum", "max", "min"

    return:
        - the average tensor after maskin out 0's
        - None if the gather resulted in an empty tensor
    """

    assert method in [
        "mean",
        "sum",
        "max",
        "min",
    ], "This function has limited capabilities [sum, mean, max, min]"
    assert type(x) is not None, "Cannot reduce a None type object"

    # wait for everyone to arrive here before gathering

    if type(x) is not torch.Tensor:
        x = torch.tensor([x])

    # verify that the tensor is on the proper device
    x = x.to(trainer.device)

    # pad across processes
    padded_x = trainer.accelerator.pad_across_processes(x, dim=0)

    # gather across all procesess
    gathered_x = trainer.accelerator.gather(padded_x)

    # mask out zeros
    masked_x = gathered_x[gathered_x != 0]

    # if the tensor is empty, warn and return None
    if len(masked_x) == 0:
        click.secho(
            f"The call to this method resulted in an empty tensor after masking out zeros. The gathered tensor was this: {gathered_x} and the original value passed was: {x}.",
            fg="red",
        )
        return None

    if method == "mean":
        return torch.mean(masked_x)
    elif method == "sum":
        return torch.sum(masked_x)
    elif method == "max":
        return torch.max(masked_x)
    elif method == "min":
        return torch.min(masked_x)


def save_trainer(
    tracker: tracker_local.Tracker,
    trainer: local_trainer.DiffusionPriorTrainer,
    is_latest: bool,
    is_best: bool,
    samples_seen: int,
    best_validation_loss: float,
):
    """
    Logs the model with an appropriate method depending on the tracker
    """
    trainer.accelerator.wait_for_everyone()

    if trainer.accelerator.is_main_process:
        click.secho(
            f"RANK:{trainer.accelerator.process_index} | Saving Model | Best={is_best} | Latest={is_latest}",
            fg="magenta",
        )

    tracker.save(
        trainer=trainer,
        is_best=is_best,
        is_latest=is_latest,
        samples_seen=int(samples_seen),
        best_validation_loss=best_validation_loss,
    )


def recall_trainer(tracker: tracker_local.Tracker, trainer: local_trainer.DiffusionPriorTrainer):
    """
    Loads the model with an appropriate method depending on the tracker
    """

    if trainer.accelerator.is_main_process:
        click.secho(f"Loading model from {type(tracker.loader).__name__}", fg="yellow")


    state_dict = tracker.recall()

    trainer.load(state_dict, strict=True)
    #trainer.load(state_dict, strict=False)

    return (
        int(state_dict.get("epoch", 0)),
        state_dict.get("best_validation_loss", 0),
        int(state_dict.get("samples_seen", 0)),
    )


# eval functions


def report_validation_loss(
    trainer: local_trainer.DiffusionPriorTrainer,
    dataloader: DataLoader,
    text_conditioned: bool,
    use_ema: bool,
    tracker: tracker_local.Tracker,
    split: str,
    tracker_folder: str,
    loss_type: str,

):
    """
    Compute the validation loss on a given subset of data.
    """

    if trainer.accelerator.is_main_process:
        click.secho(
            f"Measuring performance on {use_ema}-{split} split",
            fg="green",
            blink=True,
        )

    total_loss = torch.zeros(1, dtype=torch.float, device=trainer.device)


    for ctpa_embeddings, cxr_data, labels, _ in dataloader:
        ctpa_embeddings = ctpa_embeddings.to(trainer.device)
        cxr_data = cxr_data.to(trainer.device)
        labels = labels.to(trainer.device)

        input_args = dict(image_embed=ctpa_embeddings)

        if text_conditioned: # not the case since we're not text conditioned
            input_args = dict(**input_args, text=cxr_data)
        else:
            input_args = dict(**input_args, text_embed=cxr_data, labels=labels)

        if use_ema:
            loss = trainer.ema_diffusion_prior(**input_args)

        else:
            loss = trainer(**input_args)


        total_loss += loss


    # compute the average loss across all processes

    avg_loss = pad_gather_reduce(trainer, total_loss, method="mean")
    stats = {f"{tracker_folder}/{loss_type}-loss": avg_loss}

    # print and log results on main process
    tracker.log(stats, step=trainer.step.item() + 1)

    return avg_loss


def report_cosine_sims(
    trainer: local_trainer.DiffusionPriorTrainer,
    dataloader: DataLoader,
    text_conditioned: bool,
    tracker: tracker_local.Tracker,
    split: str,
    timesteps: int,
    tracker_folder: str,
):
    trainer.eval()
    trainer.diffusion_prior.eval()
    trainer.ema_diffusion_prior.eval()
    trainer.ema_diffusion_prior.ema_model.eval()
    if trainer.accelerator.is_main_process:
        click.secho(
            f"Measuring Cosine-Similarity on {split} split with {timesteps} timesteps",
            fg="green",
            blink=True,
        )


    for test_ctpa_embeddings, cxr_data, _ in dataloader:
        test_ctpa_embeddings = test_ctpa_embeddings.to(trainer.device)
        cxr_data = cxr_data.to(trainer.device)

        # we are text conditioned, we produce an embedding from the tokenized text
        if text_conditioned:
            cxr_embedding, cxr_encodings = trainer.embed_text(cxr_data)
            cxr_cond = dict(text_embed=cxr_embedding, text_encodings=cxr_encodings)
        else:
            cxr_embedding = cxr_data
            cxr_cond = dict(text_embed=cxr_embedding)


        # prepare the text embedding
        cxr_embed = cxr_embedding / cxr_embedding.norm(dim=1, keepdim=True)

        # prepare image embeddings
        test_ctpa_embeddings = test_ctpa_embeddings / test_ctpa_embeddings.norm(
            dim=1, keepdim=True
        )

        # predict on the unshuffled text embeddings
        predicted_ctpa_embeddings = trainer.p_sample_loop(
            test_ctpa_embeddings.shape,
            cxr_cond,
            timesteps=timesteps,
        )


        predicted_ctpa_embeddings = (
            predicted_ctpa_embeddings
            / predicted_ctpa_embeddings.norm(dim=1, keepdim=True)
        )

        # calculate similarities
        orig_sim = pad_gather_reduce(
            trainer, cos(cxr_embed, test_ctpa_embeddings), method="mean"
        )
        pred_sim = pad_gather_reduce(
            trainer, cos(cxr_embed, predicted_ctpa_embeddings), method="mean"
        )

        pred_img_sim = pad_gather_reduce(
            trainer,
            cos(test_ctpa_embeddings, predicted_ctpa_embeddings),
            method="mean",
        )


        stats = {
            f"{tracker_folder}/baseline similarity [steps={timesteps}]": orig_sim,
            f"{tracker_folder}/similarity with text [steps={timesteps}]": pred_sim,
            f"{tracker_folder}/similarity with original image [steps={timesteps}]": pred_img_sim,
            f"{tracker_folder}/difference from baseline similarity [steps={timesteps}]": pred_sim
            - orig_sim,
        }

        tracker.log(stats, step=trainer.step.item() + 1)


def eval_model(
    trainer: local_trainer.DiffusionPriorTrainer,
    dataloader: DataLoader,
    text_conditioned: bool,
    split: str,
    tracker: tracker_local.Tracker,
    use_ema: bool,
    report_cosine: bool,
    report_loss: bool,
    timesteps: List[int],
    loss_type: str = None,

):
    """
    Run evaluation on a model and track metrics

    returns: loss if requested
    """
    trainer.eval()
    trainer.diffusion_prior.eval()
    trainer.ema_diffusion_prior.eval()
    trainer.ema_diffusion_prior.ema_model.eval()


    use_ema = "ema" if use_ema else "online"
    tracker_folder = f"metrics/{use_ema}-{split}"

    # detemine if valid timesteps are passed

    min_timesteps = trainer.accelerator.unwrap_model(
        trainer.diffusion_prior
    ).sample_timesteps
    max_timesteps = trainer.accelerator.unwrap_model(
        trainer.diffusion_prior
    ).noise_scheduler.num_timesteps

    assert all_between(
        timesteps, lower_bound=min_timesteps, upper_bound=max_timesteps
    ), f"all timesteps values must be between {min_timesteps} and {max_timesteps}: got {timesteps}"

    # measure cosine metrics across various eta and timesteps

    if report_cosine:
        for timestep in timesteps:
            report_cosine_sims(
                trainer,
                dataloader=dataloader,
                text_conditioned=text_conditioned,
                tracker=tracker,
                split=split,
                timesteps=timestep,
                tracker_folder=tracker_folder)

    # measure loss on a seperate split of data

    if report_loss:
        loss = report_validation_loss(
            trainer=trainer,
            dataloader=dataloader,
            text_conditioned=text_conditioned,
            use_ema=use_ema,
            tracker=tracker,
            split=split,
            tracker_folder=tracker_folder,
            loss_type=loss_type,
        )

        return loss


# training script


def train(
    trainer: local_trainer.DiffusionPriorTrainer,
    tracker: tracker_local.Tracker,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    test_loader: DataLoader,
    config: local_train_configs.DiffusionPriorTrainConfig,
):
    # init timers
    save_timer = Timer()  # when to save
    samples_timer = Timer()  # samples/sec
    validation_profiler = Timer()  # how long is validation taking
    validation_countdown = Timer()  # when to perform evalutation

    # keep track of best validation loss

    best_validation_loss = config.train.best_validation_loss
    samples_seen = config.train.num_samples_seen

    # do training

    start_epoch = config.train.current_epoch


    for epoch in range(start_epoch, config.train.epochs):
        # if we finished out an old epoch, reset the distribution to be a full epoch
        tracker.log({"tracking/epoch": epoch}, step=trainer.step.item())

        if train_loader.dataset.get_start() > 0 and epoch == start_epoch+1:
            if trainer.accelerator.is_main_process:
                click.secho(f"Finished resumed epoch...resetting dataloader.")
            train_loader.dataset.set_start(0)


        for ct, xray, labels in train_loader:
            # setup things every step

            trainer.train()
            current_step = trainer.step.item()
            samples_timer.reset()

            # place data on device

            ct = ct.to(trainer.device)
            xray = xray.to(trainer.device)
            labels = labels.to(trainer.device)



            # pass to model

            loss = trainer(text_embed=xray, image_embed=ct, labels=labels)


            # perform backprop & apply EMA updates

            trainer.update()

            # gather info about training step

            all_loss = pad_gather_reduce(trainer, loss, method="mean")
            num_samples = pad_gather_reduce(trainer, len(xray), method="sum")
            samples_per_sec = num_samples / samples_timer.elapsed()
            samples_seen += num_samples
            ema_decay = trainer.ema_diffusion_prior.get_current_decay()

            # log

            tracker.log(
                {
                    "tracking/samples-sec": samples_per_sec,
                    "tracking/samples-seen": samples_seen,
                    "tracking/ema-decay": ema_decay,
                    f"tracking/training-{config.prior.loss_type}": all_loss,
                },
                step=current_step,
            )

            # Metric Tracking @ Timed Intervals

            eval_delta = pad_gather_reduce(
                trainer, validation_countdown.elapsed(), method="min"
            )

            if eval_delta != None and eval_delta > config.data.eval_every_seconds:
                # begin timing how long this takes
                validation_profiler.reset()

                # package kwargs for evaluation

                eval_kwargs = {
                    "trainer": trainer,
                    "tracker": tracker,
                    "text_conditioned": config.prior.condition_on_text_encodings,
                    "timesteps": config.train.eval_timesteps,
                }

                # ONLINE MODEL : COSINE : LOSS : VALIDATION SPLIT

                eval_model(
                    dataloader=eval_loader,
                    loss_type=config.prior.loss_type,
                    split="validation",
                    use_ema=False,
                    report_cosine=False,
                    report_loss=True,
                    **eval_kwargs,
                )

                # EMA MODEL : COSINE : LOSS : VALIDATION DATA

                ema_val_loss = eval_model(
                    dataloader=eval_loader,
                    loss_type=config.prior.loss_type,
                    split="validation",
                    use_ema=True,
                    report_cosine=True,
                    report_loss=True,
                    **eval_kwargs,
                )

                tracker.log(
                    {
                        "tracking/validation length (minutes)": validation_profiler.elapsed()
                                                                / 60
                    }
                )

                # check if the ema validation is the lowest seen yet

                if ema_val_loss < best_validation_loss:
                    best_validation_loss = ema_val_loss

                    #  go save the model as best

                    save_trainer(
                        trainer=trainer,
                        tracker=tracker,
                        is_best=True,
                        is_latest=False,
                        samples_seen=samples_seen,
                        epoch=epoch,
                        best_validation_loss=best_validation_loss,
                    )

                # reset timer for validaiton

                validation_countdown.reset()

            elif eval_delta is None:
                click.secho(
                    f"Error occured reading the eval time on rank: {trainer.device}",
                    fg="yellow",
                )

            # save as latest model on schedule

            save_delta = pad_gather_reduce(trainer, save_timer.elapsed(), method="min")

            if save_delta != None and save_delta >= config.train.save_every_seconds:
                save_trainer(
                    trainer=trainer,
                    tracker=tracker,
                    is_best=False,
                    is_latest=True,
                    samples_seen=samples_seen,
                    epoch=epoch,
                    best_validation_loss=best_validation_loss,
                )

                save_timer.reset()

            elif save_delta is None:
                click.secho(
                    f"Error occured reading the save time on rank: {trainer.device}",
                    fg="yellow",
                )

    # evaluate on test data

    if trainer.accelerator.is_main_process:
        click.secho(f"Starting Test", fg="red")

    # save one last time as latest before beginning validation

    save_trainer(
        tracker=tracker,
        trainer=trainer,
        is_best=False,
        is_latest=True,
        samples_seen=samples_seen,
        epoch=epoch,
        best_validation_loss=best_validation_loss,
    )

    test_loss = eval_model(
        trainer=trainer,
        dataloader=test_loader,
        text_conditioned=config.prior.condition_on_text_encodings,
        split="test",
        tracker=tracker,
        use_ema=True,
        report_cosine=False,
        report_loss=True,
        timesteps=config.train.eval_timesteps,
        loss_type=config.prior.loss_type,
    )

    if test_loss < best_validation_loss:
        best_validation_loss = test_loss

        #  go save the model as best

        save_trainer(
            trainer=trainer,
            tracker=tracker,
            is_best=True,
            is_latest=False,
            samples_seen=samples_seen,
            epoch=epoch,
            best_validation_loss=test_loss,
        )



def initialize_training(config_file, accelerator):
    """
    Parse the configuration file, and prepare everything necessary for training
    """
    torch.cuda.empty_cache()
    # load the configuration file
    if accelerator.is_main_process:
        click.secho(f"Loading configuration from {config_file}", fg="green")

    config = local_train_configs.TrainDiffusionPriorConfig.from_json_path(config_file)

    # seed

    set_seed(config.train.random_seed)

    # get a device

    device = accelerator.device

    # make the trainer (will automatically distribute if possible & configured)


    trainer: local_trainer.DiffusionPriorTrainer = make_model(
        config.prior, config.train, device, accelerator
    ).to(device)


    # create a tracker

    tracker = create_tracker(
        accelerator, config, config_file, dummy=accelerator.process_index != 0
    )

    # reload from chcekpoint

    if tracker.can_recall:
        current_epoch, best_validation_loss, samples_seen = recall_trainer(
            tracker=tracker, trainer=trainer
        )

        # display best values
        if trainer.accelerator.is_main_process:
            click.secho(f"Current Epoch: {current_epoch} | Best Val Loss: {best_validation_loss} | Samples Seen: {samples_seen}", fg="yellow")

        # update config to reflect recalled values
        config.train.num_samples_seen = samples_seen
        config.train.current_epoch = current_epoch
        config.train.best_validation_loss = best_validation_loss

    # fetch and prepare data

    if trainer.accelerator.is_main_process:
        click.secho("Grabbing data...", fg="blue", blink=True)

    trainer.accelerator.wait_for_everyone()


    # calculate start point within epoch

    trainer.accelerator.wait_for_everyone()


    train_loader = DataLoader(XrayCTPADataset(target="y_train.csv"),batch_size=config.data.batch_size)
    eval_loader = DataLoader(XrayCTPADataset(target="y_valid.csv"),batch_size=config.data.batch_size)
    test_loader = DataLoader(XrayCTPADataset(target="y_test.csv"),batch_size=config.data.batch_size)


    # start training

    if trainer.accelerator.is_main_process:
        click.secho(
            f"Beginning Prior Training : Distributed={accelerator.state.distributed_type != accelerate_dataclasses.DistributedType.NO}",
            fg="yellow",
        )
    train(
        trainer=trainer,
        tracker=tracker,
        train_loader=train_loader,
        eval_loader=eval_loader,
        test_loader=test_loader,
        config=config,
    )

    eval_kwargs = {
        "trainer": trainer,
        "tracker": tracker,
        "text_conditioned": config.prior.condition_on_text_encodings,
        "timesteps": config.train.eval_timesteps,
    }
    eval_model(
        dataloader=test_loader,
        loss_type=config.prior.loss_type,
        split="test",
        use_ema=True,
        report_cosine=True,
        report_loss=False,
        **eval_kwargs,
    )




@click.command()
@click.option("--config_file", default="configs/train_prior_config.json")
def main(config_file):
    # start HFA
    accelerator = Accelerator()
    # setup training
    initialize_training(config_file, accelerator)


if __name__ == "__main__":
    main()