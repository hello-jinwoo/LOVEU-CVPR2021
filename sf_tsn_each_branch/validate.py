network_sf.eval()
            network_tsn.eval()

            f1_results = {}
            prec_results = {}
            rec_results = {}
            val_dicts = {}
            for s in SIGMA_LIST:
                val_dict = {}
                k = 3
                gaussian_filter = torch.FloatTensor(
                                            [np.exp(-z*z/(2*s*s))/np.sqrt(2*np.pi*s*s) for z in range(-k, k+1)]
                                            ).to(device)
                gaussian_filter = gaussian_filter.unsqueeze(0).unsqueeze(0)
                gaussian_filter /= torch.max(gaussian_filter)
                gaussian_filter = gaussian_filter.repeat(1, FEATURE_LEN, 1)
                max_pooling = nn.MaxPool1d(5, stride=1, padding=2)

                for feature, filenames, durations in validation_dataloader:
                    feature = feature.to(device)
                    with torch.no_grad():
                        pred_sf = network_sf(feature[..., :FEATURE_DIM_1])
                        pred_tsn = network_tsn(feature[..., FEATURE_DIM_1:])
                        pred_sf = torch.sigmoid(pred_sf) # [BATCH_SIZE, FEATURE_LEN]
                        pred_tsn = torch.sigmoid(pred_tsn) # [BATCH_SIZE, FEATURE_LEN]
                        pred = (pred_sf + pred_tsn) / 2

                        if s > 0:
                            out = pred.unsqueeze(-1)
                            eye = torch.eye(FEATURE_LEN).to(device)
                            out = out * eye
                            out = nn.functional.conv1d(out, gaussian_filter, padding=k)
                        else:
                            out = pred.unsqueeze(1)

                        peak = (out == max_pooling(out))
                        peak[out < THRESHOLD] = False
                        peak = peak.squeeze()

                        idx = torch.nonzero(peak).cpu().numpy()
                        
                    durations = durations.numpy()

                    boundary_list = [[] for _ in range(len(out))]
                    for i, j in idx:
                        duration = durations[i]
                        first = TIME_UNIT/2
                        if first + TIME_UNIT*j < duration:
                            boundary_list[i].append(first + TIME_UNIT*j)
                    for i, boundary in enumerate(boundary_list):
                        filename = filenames[i]
                        val_dict[filename] = boundary

                val_dicts[s] = val_dict
                f1, prec, rec = validate(val_dict, fold)
                f1_results[s] = f1
                prec_results[s] = prec
                rec_results[s] = rec
                if f1 > val_max_f1:
                    val_max_f1 = f1
                    improve_flag = True
                    no_improvement_duration = 0

            print(f'epoch: {epoch+1}, f1: {f1_results}')
            print(f'epoch: {epoch+1}, precision: {prec_results}')
            print(f'epoch: {epoch+1}, recall: {rec_results}')