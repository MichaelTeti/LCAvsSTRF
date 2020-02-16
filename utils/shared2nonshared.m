function shared2nonshared(filepath, output_path, input_h, input_w)
    addpath('/OpenPV/mlab/util');

    % Check that the output directory exists and create it if not
    if ~exist(output_path, 'dir')
       mkdir(output_path)
    end

    % read the weights in and get size of weight tensor
    weight_dict = readpvpfile(filepath);
    weight_tensor = weight_dict{1, 1}.values{1, 1};
    [nxp, nyp, nfp, nf] = size(weight_tensor);

    % compute padding and extend the weight tensor to non-shared size
    x_extended = input_w + (nxp - 1); y_extended = input_h + (nyp - 1);
    weight_tensor = reshape(weight_tensor, [1, 1, nxp, nyp, nfp, nf]);  % add two extra dims for repmat
    weight_tensor = repmat(weight_tensor, [x_extended, y_extended]);  % extend along those new dims
    weight_tensor = permute(weight_tensor, [3, 4, 5, 6, 1, 2]);  % need to save fxy for this
    weight_tensor = reshape(weight_tensor, [nxp, nyp, nfp, x_extended*y_extended*nf]);
    weight_dict{1, 1}.values{1, 1} = weight_tensor;  % put weight tensor back in struct

    % save the new weight file in the output_path directory
    [root, name, ext] = fileparts(filepath);
    save_path = strcat(output_path, "/", name, ext);
    writepvpweightfile(filename=save_path,
                       data=weight_dict,
                       nxGlobalPre=x_extended,
                       nyGlobalPre=y_extended,
                       nfPre=nf,
                       nbPre=0,
                       nxGlobalPost=input_w,
                       nyGlobalPost=input_h,
                       postweightsflag=true);

end
