from datetime import datetime

import mlflow
from omegaconf import OmegaConf

from src.data import (
    create_splits,
    stateless_preprocessing,
    load_data,
    make_pipeline,
    make_test_split,
)

def main():
    cfg = OmegaConf.load('config.yaml')

    ### IMPORTS BY MODEL TYPE ###

    if cfg.model_type == 'torch':
        from src.nn import (
        create_dataloaders,
        MLP,
        create_optimizer,
        create_loss_fn,
        run_optuna_study,
        prepare_full_data,
        train_final_nn,
        final_evaluation,
        save_model
        )
    elif cfg.model_type == 'boosting':
        from src.boosting import (
        prepare_cat_columns,
        run_optuna_study,
        prepare_full_data,
        train_full_boosting,
        final_evaluation,
        save_boosting_model,
        )

    ### OPTUNA STUDY START ###

    # BOOSTING #

    if cfg.result.mode.study is True and cfg.model_type == 'boosting':
        # DATA PREPARATION
        df = load_data(cfg)
        df = stateless_preprocessing(df)
        df, test = make_test_split(cfg, df)
        pipeline = make_pipeline(cfg, df)
        df = create_splits(cfg, df) 
        df, cat_features = prepare_cat_columns(df)

        # OPTUNA STUDY BOOSTING RUN
        experiment_name = f"optuna_study_{cfg.model_class}_{datetime.now().strftime('%d%m%Y_%H%M')}"
        mlflow.set_experiment(experiment_name)

        best_cfg, num_trees = run_optuna_study(cfg, df, pipeline, experiment_name, cat_features, show_plots=False)
        print(f'Best trial numbers of trees by folds: {num_trees}')

    # TORCH #

    elif cfg.result.mode.study is True and cfg.model_type == 'torch':
        # DATA PREPARATION    
        df = load_data(cfg)
        df = stateless_preprocessing(df)
        df, test = make_test_split(cfg, df)
        pipeline = make_pipeline(cfg, df)
        df = create_splits(cfg, df)

        # OPTUNA STUDY NN RUN
        experiment_name = f"optuna_study_{cfg.model_class}_{datetime.now().strftime('%d%m%Y_%H%M')}"
        mlflow.set_experiment(experiment_name)

        best_cfg, best_num_epochs = run_optuna_study(cfg, df, pipeline, experiment_name, show_plots=False)
        print(f'Best trial numbers of epochs by folds: {best_num_epochs}')

    ### TRAINING MODEL RUN ###

    if cfg.result.mode.study is False and cfg.result.mode.train is True:
        best_cfg = cfg # uses default config for training
        experiment_name = f"train_{cfg.model_class}_{datetime.now().strftime('%d%m%Y_%H%M')}"
        mlflow.set_experiment(experiment_name) # creates mlflow experiment for full training. Else used experiment from Optuna study

    # BOOSTING #
    if cfg.result.mode.train is True and cfg.model_type == 'boosting':
        
        # DATA PREPARATION
        df = load_data(cfg)
        df = stateless_preprocessing(df)
        df, test = make_test_split(cfg, df)
        pipeline = make_pipeline(cfg, df)
        df, cat_features = prepare_cat_columns(df)

        X_test, y_test = None, None

        if test is not None:
            test, _ = prepare_cat_columns(test)
            X_train, y_train, X_test, y_test = prepare_full_data(cfg, df, pipeline, test)
        else:
            X_train, y_train = prepare_full_data(cfg, df, pipeline)

        # TRAINING FULL BOOSTING MODEL
        with mlflow.start_run(run_name="final_model"):
            model = train_full_boosting(best_cfg, X_train, y_train, cat_features)
            # evaluation
            if X_test is not None:
                metrics = final_evaluation(cfg, model, X_test, y_test)
                print(metrics)
            # save model
            save_boosting_model(cfg, model)

    # TORCH #
    if cfg.result.mode.train is True and cfg.model_type == 'torch':

        # DATA PREPARATION
        df = load_data(cfg)
        df = stateless_preprocessing(df)
        df, test = make_test_split(cfg, df)
        pipeline = make_pipeline(cfg, df)

        test_loader = None

        if test is not None:
            X_train, y_train, X_test, y_test = prepare_full_data(cfg, df, pipeline, test)
            train_loader, test_loader = create_dataloaders(best_cfg, X_train, y_train, X_test, y_test)
        else:
            X_train, y_train = prepare_full_data(cfg, df, pipeline)
            train_loader = create_dataloaders(best_cfg, X_train, y_train)

        # MODEL PREPARATION
        model = MLP(**best_cfg.nn.model)
        optimizer = create_optimizer(model, best_cfg)
        loss_fn = create_loss_fn(best_cfg)

        # FULL TRAINING
        with mlflow.start_run(run_name="final_model"):
            model, train_losses = train_final_nn(model, train_loader, optimizer, loss_fn, best_cfg)
            if test_loader is not None:
                metrics = final_evaluation(best_cfg, model, test_loader)
                print(metrics)
            save_model(model)

if __name__ == "__main__":
    main()