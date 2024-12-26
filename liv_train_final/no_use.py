
        # if epoch % 25 == 24:
        #     self_attention_model.eval()
        #     if args.pca:
        #         transform_model.eval()
        #     with torch.no_grad():
        #         if args.pca:
        #             plot_videos(args.model_name, self_attention_model, pca_text_model, pca_video_model, transform_model)
        #         else:
        #             plot_videos(args.model_name, self_attention_model)
        #     self_attention_model.train()
        #     if args.pca:
        #         transform_model.train()

        
        # if epoch % 10 == 9:
        #     self_attention_model.eval()
        #     if args.pca:
        #         transform_model.eval()
        #     with torch.no_grad():

        #         if args.pca:
        #             plot_confusion_matrix_pca(h5_file, args.model_name, "train", self_attention_model, pca_text_model, pca_video_model, transform_model)
        #             plot_confusion_matrix_pca(h5_file, args.model_name, "eval", self_attention_model, pca_text_model, pca_video_model, transform_model)
        #         else:
        #             plot_confusion_matrix_pca(h5_file, args.model_name, "train", self_attention_model)
        #             plot_confusion_matrix_pca(h5_file, args.model_name, "eval", self_attention_model)

        #         if args.pca:
        #             corr_train_dict = plot_progress_corr(h5_file, args.model_name, "train", self_attention_model, pca_text_model, pca_video_model, transform_model)
        #             corr_eval_dict = plot_progress_corr(h5_file, args.model_name, "eval", self_attention_model, pca_text_model, pca_video_model, transform_model)
        #         else:
        #             corr_train_dict = plot_progress_corr(h5_file, args.model_name, "train", self_attention_model)
        #             corr_eval_dict = plot_progress_corr(h5_file, args.model_name, "eval", self_attention_model)

        #         wandb_log = {
        #             "corr_train": corr_train_dict,
        #             "corr_eval": corr_eval_dict
        #         }
        #         wandb.log(wandb_log)

        #         if args.pca:
        #             plot_progress(h5_file, args.model_name, "train", self_attention_model, pca_text_model, pca_video_model, transform_model)
        #             plot_progress(h5_file, args.model_name, "eval", self_attention_model, pca_text_model, pca_video_model, transform_model)
        #         else:
        #             plot_progress(h5_file, args.model_name, "train", self_attention_model)
        #             plot_progress(h5_file, args.model_name, "eval", self_attention_model)


        #     self_attention_model.train()
        #     if args.pca:
        #         transform_model.train()

        # if epoch % 25 == 24:
        #     save_model_path = "/scr/jzhang96/clip_liv_models"
        #     if not os.path.exists(save_model_path):
        #         os.makedirs(save_model_path)
        #     folder_name = experiment_name
        #     if not os.path.exists(os.path.join(save_model_path, folder_name)):
        #         os.makedirs(os.path.join(save_model_path, folder_name))
        #     torch.save(self_attention_model.state_dict(), os.path.join(save_model_path, folder_name, f"model_{epoch}.pt"))
        #     if args.pca:
        #         torch.save(transform_model.state_dict(), os.path.join(save_model_path, folder_name, f"transform_{epoch}.pt"))
        #         # dump pca models
        #         pca_video_path = os.path.join(save_model_path, folder_name, f"pca_video.pkl")
        #         pca_text_path = os.path.join(save_model_path, folder_name, f"pca_text.pkl")
        #         joblib.dump(pca_video_model, pca_video_path)
        #         joblib.dump(pca_text_model, pca_text_path)
            



        