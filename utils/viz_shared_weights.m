function viz_shared_weights(checkpoint_path)
    addpath('/OpenPV/mlab/util');

    if ~exist(checkpoint_path, 'dir')
       printf('Directory given to VizSharedWeights does not exist.')
       return
    end

    if checkpoint_path(end) ~= '/'
      checkpoint_path = strcat(checkpoint_path, '/');
    end

    fpaths = dir(strcat(checkpoint_path, '*_W.pvp'));
    n_fpaths = numel(fpaths);

    for i_fpath = 1:n_fpaths
        fpath = strcat(checkpoint_path, fpaths(i_fpath, 1).name);
        w = readpvpfile(fpath);
        w = w{1, 1}.values{1, 1};

        nxp = size(w, 1); nyp = size(w, 2); nfp = size(w, 3); nf = size(w, 4);
        grid_dim = ceil(sqrt(nf));
        grid_h = grid_dim * nyp; grid_w = grid_dim * nxp;
        grid = zeros(grid_h, grid_w, nfp);

        for i = 1:grid_dim
            for j = 1:grid_dim
                if (i-1)*grid_dim+j <= nf
                    patch = w(:, :, :, (i-1)*grid_dim+j);

                    if ndims(patch) == 2
                        patch = transpose(patch);
                        patch = patch - min(min(patch));
                        patch = patch / (max(max(patch)) + 1e-6);
                    elseif ndims(patch) == 3 & size(patch, 3) == 3
                        patch = permute(patch, [2, 1, 3]);
                        patch = patch - min(min(min(patch)));
                        patch = patch / (max(max(max(patch))) + 1e-6);
                    end

                    grid((i-1)*nyp+1:(i-1)*nyp+nyp, (j-1)*nxp+1:(j-1)*nxp+nxp, :) = patch;
                end  % if (i-1)
            end  % for j = 1:grid_dim
        end  % for i = 1:grid_dim

        if n_fpaths == 1
            imwrite(grid, 'weights.gif')
        else
            if i_fpath == 1
                imwrite(grid, 'weights.gif', 'gif', 'writemode', 'overwrite', 'Loopcount', inf, 'DelayTime', 0.5);
            else
                imwrite(grid, 'weights.gif', 'gif', 'writemode', 'append', 'DelayTime', 0.1);
            end
        end

    end
