#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>

typedef void* CoreMLModelRef;

CoreMLModelRef coreml_load_model(const char* model_path) {
    @autoreleasepool {
        NSLog(@"[CoreML] Loading model from: %s", model_path);
        NSString *path = [NSString stringWithUTF8String:model_path];
        NSURL *modelURL = [NSURL fileURLWithPath:path];
        
        NSError *error = nil;
        NSURL *compiledModelURL = modelURL;
        
        if ([path hasSuffix:@".mlpackage"] || [path hasSuffix:@".mlmodel"]) {
            NSLog(@"[CoreML] Detected uncompiled model, compiling...");
            compiledModelURL = [MLModel compileModelAtURL:modelURL error:&error];
            
            if (error) {
                NSLog(@"[CoreML] Error compiling model: %@", error.localizedDescription);
                return NULL;
            }
            
            NSLog(@"[CoreML] Model compiled successfully to: %@", compiledModelURL.path);
        }
        
        NSLog(@"[CoreML] Loading compiled model...");
        MLModel *model = [MLModel modelWithContentsOfURL:compiledModelURL error:&error];
        
        if (error) {
            NSLog(@"[CoreML] Error loading model: %@", error.localizedDescription);
            return NULL;
        }
        
        NSLog(@"[CoreML] Model loaded successfully");
        NSLog(@"[CoreML] Model description: %@", model.modelDescription);
        NSLog(@"[CoreML] Input features: %@", model.modelDescription.inputDescriptionsByName.allKeys);
        NSLog(@"[CoreML] Output features: %@", model.modelDescription.outputDescriptionsByName.allKeys);
        
        return (__bridge_retained void*)model;
    }
}

int coreml_predict_multiarray(CoreMLModelRef model_ref,
                               const char* input_name,
                               const float* input_data,
                               size_t input_size,
                               const char* output_name,
                               float* output_data,
                               size_t output_size) {
    @autoreleasepool {
        NSLog(@"[CoreML] Starting prediction...");
        NSLog(@"[CoreML] Input name: %s, size: %zu", input_name, input_size);
        NSLog(@"[CoreML] Output name: %s, size: %zu", output_name, output_size);
        
        if (!model_ref) {
            NSLog(@"[CoreML] Error: NULL model reference");
            return -1;
        }
        
        MLModel *model = (__bridge MLModel*)model_ref;
        
        NSError *error = nil;
        MLModelDescription *description = model.modelDescription;
        
        NSString *inputNameStr = [NSString stringWithUTF8String:input_name];
        MLFeatureDescription *inputDesc = description.inputDescriptionsByName[inputNameStr];
        
        if (!inputDesc) {
            NSLog(@"[CoreML] Error: Input feature '%@' not found in model", inputNameStr);
            NSLog(@"[CoreML] Available inputs: %@", description.inputDescriptionsByName.allKeys);
            return -2;
        }
        
        NSArray<NSNumber *> *shape = inputDesc.multiArrayConstraint.shape;
        NSLog(@"[CoreML] Input shape: %@", shape);
        
        MLMultiArray *inputArray = [[MLMultiArray alloc] initWithShape:shape
                                                               dataType:MLMultiArrayDataTypeFloat32
                                                                  error:&error];
        if (error) {
            NSLog(@"[CoreML] Error creating input array: %@", error.localizedDescription);
            return -3;
        }
        
        float *dataPtr = (float *)inputArray.dataPointer;
        size_t copySize = MIN(input_size, inputArray.count);
        memcpy(dataPtr, input_data, copySize * sizeof(float));
        
        NSLog(@"[CoreML] Creating feature provider...");
        MLDictionaryFeatureProvider *inputProvider = [[MLDictionaryFeatureProvider alloc]
            initWithDictionary:@{inputNameStr: inputArray}
            error:&error];
        
        if (error) {
            NSLog(@"[CoreML] Error creating feature provider: %@", error.localizedDescription);
            return -4;
        }
        
        NSLog(@"[CoreML] Running prediction...");
        id<MLFeatureProvider> prediction = [model predictionFromFeatures:inputProvider error:&error];
        
        if (error) {
            NSLog(@"[CoreML] Prediction error: %@", error.localizedDescription);
            return -5;
        }
        
        NSLog(@"[CoreML] Prediction completed successfully");
        NSLog(@"[CoreML] Available output features: %@", prediction.featureNames);
        
        NSString *outputNameStr = [NSString stringWithUTF8String:output_name];
        MLFeatureValue *outputFeature = [prediction featureValueForName:outputNameStr];
        
        if (!outputFeature) {
            NSLog(@"[CoreML] Error: Output feature '%@' not found", outputNameStr);
            NSLog(@"[CoreML] Available outputs: %@", prediction.featureNames);
            return -6;
        }
        
        if (outputFeature.type != MLFeatureTypeMultiArray) {
            NSLog(@"[CoreML] Error: Output feature '%@' is not a multi-array (type: %ld)", 
                  outputNameStr, (long)outputFeature.type);
            return -6;
        }
        
        MLMultiArray *outputArray = outputFeature.multiArrayValue;
        NSLog(@"[CoreML] Output array shape: %@, count: %ld", outputArray.shape, (long)outputArray.count);
        
        float *outputPtr = (float *)outputArray.dataPointer;
        size_t outputCopySize = MIN(output_size, outputArray.count);
        memcpy(output_data, outputPtr, outputCopySize * sizeof(float));
        
        NSLog(@"[CoreML] Copied %zu values to output buffer", outputCopySize);
        return 0;
    }
}

// FIXME: This model requires MLState inputs that can't be properly initialized.
// The model was exported with coremltools 8.0 using stateful=True, but the
// state objects (keyCache, valueCache) need to be passed as inputs.
// 
// Workaround options:
// 1. Re-export the model with state_management="automatic" in coremltools
// 2. Export with stateful=False (no KV cache, slower but simpler)
// 3. Use a different deployment method (e.g., ONNX Runtime, llama.cpp)
//
// For now, this will fail on first prediction.
static NSMutableDictionary *global_state_dict = nil;

int coreml_predict_stateful(CoreMLModelRef model_ref,
                             const int32_t* input_ids,
                             size_t input_ids_size,
                             const float* causal_mask,
                             size_t mask_size,
                             float* output_data,
                             size_t output_size) {
    @autoreleasepool {
        if (!model_ref) {
            NSLog(@"[CoreML] Error: NULL model reference");
            return -1;
        }
        
        MLModel *model = (__bridge MLModel*)model_ref;
        NSError *error = nil;
        
        NSArray<NSNumber *> *inputIdsShape = @[@1, @(input_ids_size)];
        MLMultiArray *inputIdsArray = [[MLMultiArray alloc] initWithShape:inputIdsShape
                                                                  dataType:MLMultiArrayDataTypeInt32
                                                                     error:&error];
        if (error) {
            NSLog(@"[CoreML] Error creating inputIds array: %@", error.localizedDescription);
            return -2;
        }
        
        int32_t *inputIdsPtr = (int32_t *)inputIdsArray.dataPointer;
        memcpy(inputIdsPtr, input_ids, input_ids_size * sizeof(int32_t));
        
        size_t mask_dim = (size_t)sqrt((double)mask_size);
        NSArray<NSNumber *> *maskShape = @[@1, @1, @(mask_dim), @(mask_dim)];
        MLMultiArray *causalMaskArray = [[MLMultiArray alloc] initWithShape:maskShape
                                                                    dataType:MLMultiArrayDataTypeFloat16
                                                                       error:&error];
        if (error) {
            NSLog(@"[CoreML] Error creating causalMask array: %@", error.localizedDescription);
            return -3;
        }
        
        __fp16 *maskPtr = (__fp16 *)causalMaskArray.dataPointer;
        for (size_t i = 0; i < mask_size; i++) {
            maskPtr[i] = (__fp16)causal_mask[i];
        }
        
        NSMutableDictionary *inputDict = [NSMutableDictionary dictionaryWithDictionary:@{
            @"inputIds": inputIdsArray,
            @"causalMask": causalMaskArray
        }];
        
        if (global_state_dict != nil && global_state_dict.count > 0) {
            NSLog(@"[CoreML] Adding cached state inputs: %lu entries", (unsigned long)global_state_dict.count);
            [inputDict addEntriesFromDictionary:global_state_dict];
        }
        
        MLDictionaryFeatureProvider *inputProvider = [[MLDictionaryFeatureProvider alloc]
            initWithDictionary:inputDict
            error:&error];
        
        if (error) {
            NSLog(@"[CoreML] Error creating feature provider: %@", error.localizedDescription);
            return -4;
        }
        
        MLPredictionOptions *options = [[MLPredictionOptions alloc] init];
        
        if (global_state_dict == nil) {
            global_state_dict = [NSMutableDictionary dictionary];
            NSLog(@"[CoreML] First prediction - will capture output states");
        }
        
        options.outputBackings = global_state_dict;
        
        id<MLFeatureProvider> prediction = [model predictionFromFeatures:inputProvider
                                                                 options:options
                                                                   error:&error];
        
        if (error) {
            NSLog(@"[CoreML] Prediction error: %@", error.localizedDescription);
            return -5;
        }
        
        NSLog(@"[CoreML] Prediction output features: %@", prediction.featureNames);
        for (NSString *name in prediction.featureNames) {
            MLFeatureValue *val = [prediction featureValueForName:name];
            NSLog(@"[CoreML]   - %@: type=%ld", name, (long)val.type);
            
            if (val.type == 7) {
                NSLog(@"[CoreML]   Storing state: %@", name);
                global_state_dict[name] = val;
            }
        }
        
        MLFeatureValue *outputFeature = [prediction featureValueForName:@"logits"];
        
        if (!outputFeature || outputFeature.type != MLFeatureTypeMultiArray) {
            NSLog(@"[CoreML] Error: Invalid logits output");
            return -6;
        }
        
        MLMultiArray *outputArray = outputFeature.multiArrayValue;
        
        if (outputArray.dataType == MLMultiArrayDataTypeFloat16) {
            __fp16 *outputPtr = (__fp16 *)outputArray.dataPointer;
            size_t copySize = MIN(output_size, outputArray.count);
            for (size_t i = 0; i < copySize; i++) {
                output_data[i] = (float)outputPtr[i];
            }
        } else if (outputArray.dataType == MLMultiArrayDataTypeFloat32) {
            float *outputPtr = (float *)outputArray.dataPointer;
            size_t copySize = MIN(output_size, outputArray.count);
            memcpy(output_data, outputPtr, copySize * sizeof(float));
        } else {
            NSLog(@"[CoreML] Error: Unsupported output data type");
            return -7;
        }
        
        return 0;
    }
}

void coreml_free_model(CoreMLModelRef model_ref) {
    if (model_ref) {
        CFRelease(model_ref);
    }
}

int coreml_get_input_shape(CoreMLModelRef model_ref,
                            const char* input_name,
                            int64_t* shape_out,
                            size_t max_dims) {
    @autoreleasepool {
        if (!model_ref) {
            return -1;
        }
        
        MLModel *model = (__bridge MLModel*)model_ref;
        NSString *inputNameStr = [NSString stringWithUTF8String:input_name];
        
        MLModelDescription *description = model.modelDescription;
        MLFeatureDescription *inputDesc = description.inputDescriptionsByName[inputNameStr];
        
        if (!inputDesc) {
            return -2;
        }
        
        NSArray<NSNumber *> *shape = inputDesc.multiArrayConstraint.shape;
        size_t dims = MIN(shape.count, max_dims);
        
        for (size_t i = 0; i < dims; i++) {
            shape_out[i] = [shape[i] longLongValue];
        }
        
        return (int)dims;
    }
}
