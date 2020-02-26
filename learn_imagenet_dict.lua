-- Path to OpenPV. Replace this with an absolute path.
package.path = package.path .. ";" .. "/OpenPV/parameterWrapper/?.lua";
local pv = require "PVModule";


-- Input Image Vars
local imageInputPath        = "/home/mteti/Allen/LCAvsSTRF/filenames_test.txt";
local inputFeatures         = 1;
local inputHeight           = 32;
local inputWidth            = 64;
local nbatch                = 10;
local numImages             = 315176;


-- Model Vars
local AMax                  = infinity;
local AMin                  = 0;
local basePhase             = 2;
local dictionarySize        = 1000;
local displayMultiple       = 1;
local displayPeriod         = 4000;
local initFromCkpt          = false;
local initFromCkptPath      = "runs/run17_LCA_init_rand/Checkpoints/Checkpoint";
local initFromFile          = true;
local initFromFilePath      = "imagenet_psx17_psy17_inH32_inW64_dsize1000/run12_softThresh_LCA_Tau4000_training/Checkpoints/Checkpoint00002000/";
local initFromFilePrefix    = nil;
local learningRate          = 0.1;
local modelType             = "LCA";
local momentumTau           = 100;
local numEpochs             = 1;
local patchSizeX            = 17;
local patchSizeY            = 17;
local plasticity            = false;
local sharedWeights         = true;
local startFrame            = 0;
local startTime             = 0;
local stopTime              = 4000; --math.ceil(numImages / nbatch) * displayPeriod *
                                --displayMultiple * numEpochs;
local strideX               = 1;
local strideY               = 1;
local temporalPatchSize     = 9;
local threshType            = "soft";
local timeConstantTau       = 3000;
local useGPU                = true;
local VThresh               = 0.125;


--Probes and Checkpointing
local adaptiveThreshProbe   = false;
local checkpointPeriod      = 50; --displayPeriod * displayMultiple;
local deleteOldCheckpoints  = false;
local energyProbe           = false;
local error2ModelWriteStep  = -1;
local errorWriteStep        = -1;
local firmThreshProbe       = false;
local inputWriteStep        = 50;
local l2Probe               = false;
local model2ErrorWriteStep  = -1;
local model2ReconWriteStep  = -1;
local modelWriteStep        = 50;
local numCheckpointsKept    = 2;
local runNote               = "VThresh0.125_Test";
local runVersion            = 19;


local outputPath            = "runs/run" .. runVersion .. "_" .. modelType;

if runNote then
    outputPath = outputPath .. "_" .. runNote;
end


if threshType == "soft" then
    VWidth = infinity;
elseif threshType == "firm" then
    VWidth = VThresh;
end


if initFromFile and string.sub(initFromFilePath, -1) ~= "/" then
    initFromFilePath = initFromFilePath .. "/";
end



local pvParams = {
    column = {
        groupType                           = "HyPerCol";
        startTime                           = startTime;
        dt                                  = 1;
        stopTime                            = stopTime;
        progressInterval                    = checkpointPeriod;
        writeProgressToErr                  = true;
        verifyWrites                        = false;
        outputPath                          = outputPath;
        printParamsFilename                 = "model.params";
        randomSeed                          = 10000000;
        nx                                  = inputWidth;
        ny                                  = inputHeight;
        nbatch                              = nbatch;
        initializeFromCheckpointDir         = nil;
        checkpointWrite                     = true;
        checkpointWriteDir                  = outputPath .. "/Checkpoints";
        checkpointWriteTriggerMode          = "step";
        checkpointWriteStepInterval         = checkpointPeriod;
        deleteOlderCheckpoints              = deleteOldCheckpoints;
        numCheckpointsKept                  = numCheckpointsKept;
        suppressNonplasticCheckpoints       = false;
        writeTimescales                     = true;
        errorOnNotANumber                   = false;
    }
}

if initFromCkpt then
   pvParams.column.initializeFromCheckpointDir = initFromCkptPath;
end


if adaptiveThreshProbe then
    pv.addGroup(pvParams,
        "AdaptiveTimeScales", {
            groupType                           = "LogTimeScaleProbe";
            targetName                          = "EnergyProbe";
            message                             = NULL;
            textOutputFlag                      = true;
            probeOutputFile                     = "AdaptiveTimeScales.txt";
            triggerLayerName                    = "Frame" .. 0;
            triggerOffset                       = 0;
            baseMax                             = 1.1;
            baseMin                             = 1.0;
            tauFactor                           = 0.025;
            growthFactor                        = 0.025;
            logThresh                           = 10.0;
            logSlope                            = 0.01;
        }
    )
end  -- if adaptiveThreshProbe


for i_frame = 1, temporalPatchSize do
    local start_frame_index_array = {};

    for i_batch = 1,nbatch do
        start_frame_index_array[i_batch] = startFrame + i_frame-1;
    end

    pv.addGroup(pvParams,
        "Frame" .. i_frame-1, {
            groupType                       = "ImageLayer";
    	    nxScale                         = 1;
    	    nyScale                         = 1;
    	    nf                              = inputFeatures;
    	    phase                           = 1;
    	    mirrorBCflag                    = true;
    	    writeStep                       = inputWriteStep;
    	    initialWriteTime                = inputWriteStep;
    	    sparseLayer                     = false;
    	    updateGpu                       = false;
    	    dataType                        = nil;
    	    inputPath                       = imageInputPath;
    	    offsetAnchor                    = "cc";
    	    offsetX                         = 0;
    	    offsetY                         = 0;
    	    writeImages                     = 0;
    	    inverseFlag                     = false;
    	    normalizeLuminanceFlag          = true;
    	    normalizeStdDev                 = true;
    	    jitterFlag                      = 0;
    	    useInputBCflag                  = false;
    	    padValue                        = 0;
    	    autoResizeFlag                  = true;
    	    aspectRatioAdjustment           = "crop";
    	    interpolationMethod             = "bicubic";
    	    displayPeriod                   = displayPeriod * displayMultiple;
    	    batchMethod                     = "byList";
    	    start_frame_index               = start_frame_index_array;
    	    writeFrameToTimestamp           = true;
    	    resetToStartOnLoop              = false;
    	    initializeFromCheckpointFlag    = false;
        }
    )

end -- i_frame


if modelType == "LCA" then
    modelPrefix = "S";
elseif modelType == "STRF" then
    modelPrefix = "H";
end

local modelIndex = "1";
local modelLayer0 = modelPrefix .. modelIndex;
local modelLayer = modelLayer0;
local inputLayer0 = "Frame0";


if modelType == "LCA" then
    pv.addGroup(pvParams,
        modelLayer, {
            groupType                           = "HyPerLCALayer";
            nxScale                             = 1;
            nyScale                             = 1;
            nf                                  = dictionarySize;
            phase                               = basePhase+1;
            mirrorBCflag                        = false;
            valueBC                             = 0;
            initializeFromCheckpointFlag        = false;
            InitVType                           = "ConstantV";
            valueV                              = 0.0*VThresh;
            triggerLayerName                    = NULL;
            writeStep                           = modelWriteStep;
            initialWriteTime                    = modelWriteStep;
            sparseLayer                         = true;
            writeSparseValues                   = true;
            updateGpu                           = useGPU;
            dataType                            = nil;
            VThresh                             = VThresh;
            AMin                                = AMin;
            AMax                                = AMax;
            AShift                              = 0;
            VWidth                              = VWidth;
            clearGSynInterval                   = 0;
            timeConstantTau                     = timeConstantTau;
            selfInteract                        = true;
            adaptiveTimeScaleProbe              = nil;
        }
    )

elseif modelType == "STRF" then
    pv.addGroup(pvParams,
        modelLayer, {
            groupType                        = "HyPerLayer";
            nxScale                          = 1;
            nyScale                          = 1;
            nf                               = dictionarySize;
            phase                            = basePhase;
            mirrorBCflag                     = false;
            valueBC                          = 0;
            initializeFromCheckpointFlag     = false;
            InitVType                        = "ZeroV";
            triggerLayerName                 = "Frame0";
            triggerOffset                    = 1;
            triggerBehavior                  = "updateOnlyOnTrigger";
            writeStep                        = modelWriteStep;
            initialWriteTime                 = modelWriteStep;
            sparseLayer                      = false;
            updateGpu                        = false;
            dataType                         = nil;
        }
    )

end  -- if modelType == "LCA"


if ATSProbe then
    pvParams[modelLayer].adaptiveTimeScaleProbe       = "AdaptiveTimeScales";
end  -- if ATSProbe


if initFromCkpt then
    pvParams[modelLayer].initializeFromCheckpointFlag = true;
end


if firmThreshProbe then
    pv.addGroup(pvParams,
        modelLayer .. "FirmThreshProbe", {
            groupType                       = "FirmThresholdCostFnLCAProbe";
            targetLayer                     = modelLayer;
            message                         = NULL;
            textOutputFlag                  = true;
            probeOutputFile                 = modelLayer .. "FirmThreshProbe.txt";
            triggerLayerName                = NULL;
            energyProbe                     = "EnergyProbe";
            maskLayer                       = NULL;
        }
    )
end  -- if firmThreshProbe


for i_frame = 1, temporalPatchSize do

    inputLayer                              = "Frame" .. i_frame-1;
    errorLayer                              = inputLayer .. "ReconError";
    reconLayer                              = inputLayer .. "Recon";


    if l2Probe then
        pv.addGroup(pvParams,
            errorLayer .. "L2Probe", {
                groupType                   = "L2NormProbe";
                targetLayer                 = errorLayer;
                message                     = nil;
                textOutputFlag              = true;
                probeOutputFile             = errorLayer .. "L2Probe.txt";
                energyProbe                 = "EnergyProbe";
                coefficient                 = 0.5;
                maskLayerName               = nil;
                exponent                    = 2;
            }
        )
    end  -- if L2Probe



    ------------------------------- LCA MODEL ---------------------------------
    if modelType == "LCA" then
        --Error layer
        pv.addGroup(pvParams,
            errorLayer, {
                groupType                        = "HyPerLayer";
                nxScale                          = 1;
                nyScale                          = 1;
                nf                               = inputFeatures;
                phase                            = basePhase;
                mirrorBCflag                     = false;
                valueBC                          = 0;
                -- initializeFromCheckpointFlag     = false;
                InitVType                        = "ZeroV";
                triggerLayerName                 = NULL;
                writeStep                        = errorWriteStep;
                initialWriteTime                 = errorWriteStep;
                sparseLayer                      = false;
                updateGpu                        = false;
                dataType                         = nil;
            }
        )


        --Recon layer
        pv.addGroup(pvParams,
            reconLayer, pvParams[errorLayer], {
                phase = basePhase + 2;
            }
        )


        pv.addGroup(pvParams,
            inputLayer .. "To" .. errorLayer, {
                groupType                        = "IdentConn";
                preLayerName                     = inputLayer;
                postLayerName                    = errorLayer;
                channelCode                      = 0;
                scale                            = weightInit;
                delay                            = {0.000000};
            }
        )


        pv.addGroup(pvParams,
            reconLayer .. "To" .. errorLayer, {
                groupType                        = "IdentConn";
                preLayerName                     = reconLayer;
                postLayerName                    = errorLayer;
                channelCode                      = 1;
                delay                            = {0.000000};
            }
        )


        pv.addGroup(pvParams,
            errorLayer .. "To" .. modelLayer, {
                groupType                        = "TransposeConn";
                preLayerName                     = errorLayer;
                postLayerName                    = modelLayer;
                channelCode                      = 0;
                delay                            = {0.000000};
                convertRateToSpikeCount          = false;
                receiveGpu                       = useGPU;
                updateGSynFromPostPerspective    = true;
                pvpatchAccumulateType            = "convolve";
                writeStep                        = error2ModelWriteStep;
                writeCompressedCheckpoints       = false;
                selfFlag                         = false;
                gpuGroupIdx                      = -1;
                originalConnName                 = modelLayer0 .. "To" .. inputLayer .. "Recon" .. "Error";
            }
        )


        pv.addGroup(pvParams,
            modelLayer .. "To" .. reconLayer, {
                groupType                       = "CloneConn";
                preLayerName                    = modelLayer;
                postLayerName                   = reconLayer;
                channelCode                     = 0;
                writeStep                       = model2ReconWriteStep;
                delay                           = {0.000000};
                convertRateToSpikeCount         = false;
                receiveGpu                      = false;
                updateGSynFromPostPerspective   = false;
                pvpatchAccumulateType           = "convolve";
                writeCompressedCheckpoints      = false;
                selfFlag                        = false;
                originalConnName                = modelLayer0 .. "To" .. inputLayer .. "Recon" .. "Error";
            }
        )

       --plastic connections

        pv.addGroup(pvParams,
            modelLayer .. "To" .. errorLayer, {
                groupType                       = "MomentumConn";
                preLayerName                    = modelLayer;
                postLayerName                   = errorLayer;
                channelCode                     = -1;
                delay                           = {0.000000};
                numAxonalArbors                 = 1;
                convertRateToSpikeCount         = false;
                receiveGpu                      = false;
                sharedWeights                   = sharedWeights;
                initializeFromCheckpointFlag    = false;
                triggerLayerName                = "Frame" .. 0;
                triggerOffset                   = 1;
                updateGSynFromPostPerspective   = false;
                pvpatchAccumulateType           = "convolve";
                writeStep                       = model2ErrorWriteStep;
                initialWriteTime                = model2ErrorWriteStep;
                writeCompressedCheckpoints      = false;
                selfFlag                        = false;
                shrinkPatches                   = false;
                normalizeMethod                 = "normalizeL2";
                strength                        = 1;
                normalizeArborsIndividually     = false;
                normalizeOnInitialize           = true;
                normalizeOnWeightUpdate         = true;
                rMinX                           = 0;
                rMinY                           = 0;
                nonnegativeConstraintFlag       = false;
                normalize_cutoff                = 0;
                normalizeFromPostPerspective    = false;
                minL2NormTolerated              = 0;
                keepKernelsSynchronized         = true;
                useMask                         = false;
                normalizeDw                     = true;
                timeConstantTau                 = momentumTau;
                momentumMethod                  = "viscosity";
                momentumDecay                   = 0;
                nxp                             = patchSizeX;
                nyp                             = patchSizeY;
                plasticityFlag                  = plasticity;
                weightInitType                  = "UniformRandomWeight";
                initWeightsFile                 = nil;
                wMinInit                        = -1.0;
                wMaxInit                        = 1.0;
                sparseFraction                  = 0.9;
                dWMax                           = 0.1;
            }
        )


        if initFromCkpt then
            pvParams[modelLayer .. "To" .. errorLayer].initializeFromCheckpointFlag = true;
        end


        if initFromFile then
            if initFromFilePrefix then
                filePath = initFromFilePath .. initFromFilePrefix .. "To" .. inputLayer .. "ReconError_W.pvp";
            else
                filePath = initFromFilePath .. modelLayer .. "To" .. inputLayer .. "ReconError_W.pvp";
            end

            pvParams[modelLayer .. "To" .. errorLayer].weightInitType  = "FileWeight";
            pvParams[modelLayer .. "To" .. errorLayer].initWeightsFile = filePath;
        end


        if i_frame > 1 then
            pvParams[modelLayer .. "To" .. errorLayer].normalizeMethod    = "normalizeGroup";
            pvParams[modelLayer .. "To" .. errorLayer].normalizeGroupName = modelLayer0 .. "To" .. inputLayer0 .. "Recon" .. "Error";

            if not initFromCkpt and not initFromFile then
                pvParams[modelLayer .. "To" .. errorLayer].wMinInit     = 0;
                pvParams[modelLayer .. "To" .. errorLayer].wMaxInit     = 0;
            end
        end


    ------------------------------ STRF MODEL ---------------------------------
    elseif modelType == "STRF" then
        pv.addGroup(pvParams,
            modelLayer .. "To" .. inputLayer, {
                groupType                        = "HyPerConn";
                preLayerName                     = modelLayer;
                postLayerName                    = inputLayer;
                channelCode                      = -1;
                delay                            = {0.000000};
                convertRateToSpikeCount          = false;
                receiveGpu                       = false;
                updateGSynFromPostPerspective    = false;
                pvpatchAccumulateType            = "convolve";
                writeStep                        = -1;
                writeCompressedCheckpoints       = false;
                selfFlag                         = false;
                gpuGroupIdx                      = -1;
                weightInitType                   = "UniformRandomWeight";
                initWeightsFile                  = nil;
                nxp                              = patchSizeX;
                nyp                              = patchSizeY;
                normalizeMethod                  = "none";
                plasticityFlag                   = plasticity;
                sharedWeights                    = sharedWeights;
            }
        )

        pv.addGroup(pvParams,
            inputLayer .. "To" .. modelLayer, {
                groupType                        = "TransposeConn";
                preLayerName                     = inputLayer;
                postLayerName                    = modelLayer;
                channelCode                      = 0;
                delay                            = {0.000000};
                convertRateToSpikeCount          = false;
                receiveGpu                       = useGPU;
                updateGSynFromPostPerspective    = true;
                pvpatchAccumulateType            = "convolve";
                writeStep                        = -1;
                writeCompressedCheckpoints       = false;
                selfFlag                         = false;
                gpuGroupIdx                      = -1;
                normalizeMethod                  = "none";
                plasticityFlag                   = plasticity;
                originalConnName                 = modelLayer .. "To" .. inputLayer;
            }
        )


        if initFromCkpt then
            pvParams[modelLayer .. "To" .. inputLayer].initializeFromCheckpointFlag = true;
        end


        if initFromFile then
            if initFromFilePrefix then
                filePath = initFromFilePath .. initFromFilePrefix .. "To" .. inputLayer .. "ReconError_W.pvp";
            else
                filePath = initFromFilePath .. modelLayer .. "To" .. inputLayer .. "ReconError_W.pvp";
            end

            pvParams[modelLayer .. "To" .. inputLayer].weightInitType  = "FileWeight";
            pvParams[modelLayer .. "To" .. inputLayer].initWeightsFile = filePath;
        end

    end  -- if modelType == "LCA"

end -- i_frame


if energyProbe then
    pv.addGroup(pvParams,
        "EnergyProbe", {
            groupType                       = "ColumnEnergyProbe";
	        message                         = nil;
	        textOutputFlag                  = true;
	        probeOutputFile                 = "EnergyProbe.txt";
	        triggerLayerName                = nil;
	        energyProbe                     = nil;
        }
    )
end  -- if energyProbe



pv.printConsole(pvParams)
