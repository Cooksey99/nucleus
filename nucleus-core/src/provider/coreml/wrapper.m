#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

typedef void *CoreMLModelRef;

CoreMLModelRef coreml_load_model(const char *model_path) {
  @autoreleasepool {
    NSLog(@"[CoreML] Loading model from: %s", model_path);
    NSString *path = [NSString stringWithUTF8String:model_path];
    NSURL *modelURL = [NSURL fileURLWithPath:path];

    NSError *error = nil;
    NSURL *compiledModelURL = modelURL;

    // Compile .mlmodel / .mlpackage to .mlmodelc.
    if ([path hasSuffix:@".mlpackage"] || [path hasSuffix:@".mlmodel"]) {
      NSLog(@"[CoreML] Detected uncompiled model, compiling...");
      compiledModelURL = [MLModel compileModelAtURL:modelURL error:&error];

      if (error || !compiledModelURL) {
        NSLog(@"[CoreML] Error compiling model: %@",
              error.localizedDescription);
        return NULL;
      }

      NSLog(@"[CoreML] Model compiled successfully to: %@",
            compiledModelURL.path);
    }

    NSLog(@"[CoreML] Loading compiled model...");

    MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
    MLModel *model = [MLModel modelWithContentsOfURL:compiledModelURL
                                       configuration:config
                                               error:&error];

    if (error || !model) {
      NSLog(@"[CoreML] Error loading model: %@", error.localizedDescription);
      return NULL;
    }

    NSLog(@"[CoreML] Model loaded successfully");
    NSLog(@"[CoreML] Model description: %@", model.modelDescription);
    NSLog(@"[CoreML] Input features: %@",
          model.modelDescription.inputDescriptionsByName.allKeys);
    NSLog(@"[CoreML] Output features: %@",
          model.modelDescription.outputDescriptionsByName.allKeys);

    return (__bridge_retained void *)model;
  }
}

int coreml_predict_multiarray(CoreMLModelRef model_ref, const char *input_name,
                              const float *input_data, size_t input_size,
                              const char *output_name, float *output_data,
                              size_t output_size) {
  @autoreleasepool {
    if (!model_ref) {
      NSLog(@"[CoreML] Error: NULL model reference");
      return -1;
    }

    MLModel *model = (__bridge MLModel *)model_ref;
    NSError *error = nil;

    NSString *inputNameStr = [NSString stringWithUTF8String:input_name];
    MLModelDescription *description = model.modelDescription;
    MLFeatureDescription *inputDesc =
        description.inputDescriptionsByName[inputNameStr];

    if (!inputDesc) {
      NSLog(@"[CoreML] Error: Input feature '%@' not found in model",
            inputNameStr);
      return -2;
    }

    NSArray<NSNumber *> *shape = inputDesc.multiArrayConstraint.shape;
    MLMultiArray *inputArray =
        [[MLMultiArray alloc] initWithShape:shape
                                   dataType:MLMultiArrayDataTypeFloat32
                                      error:&error];
    if (error || !inputArray) {
      NSLog(@"[CoreML] Error creating input array: %@",
            error.localizedDescription);
      return -3;
    }

    float *dataPtr = (float *)inputArray.dataPointer;
    size_t copySize = input_size < (size_t)inputArray.count
                          ? input_size
                          : (size_t)inputArray.count;
    memcpy(dataPtr, input_data, copySize * sizeof(float));

    MLDictionaryFeatureProvider *inputProvider =
        [[MLDictionaryFeatureProvider alloc]
            initWithDictionary:@{inputNameStr : inputArray}
                         error:&error];
    if (error || !inputProvider) {
      NSLog(@"[CoreML] Error creating feature provider: %@",
            error.localizedDescription);
      return -4;
    }

    id<MLFeatureProvider> prediction =
        [model predictionFromFeatures:inputProvider
                                error:&error]; // [web:14][page:1]

    if (error || !prediction) {
      NSLog(@"[CoreML] Prediction error: %@", error.localizedDescription);
      return -5;
    }

    NSString *outputNameStr = [NSString stringWithUTF8String:output_name];
    MLFeatureValue *outputFeature =
        [prediction featureValueForName:outputNameStr];
    if (!outputFeature || outputFeature.type != MLFeatureTypeMultiArray) {
      NSLog(@"[CoreML] Error: Output '%@' invalid", outputNameStr);
      return -6;
    }

    MLMultiArray *outputArray = outputFeature.multiArrayValue;
    float *outputPtr = (float *)outputArray.dataPointer;
    size_t outputCopySize = output_size < (size_t)outputArray.count
                                ? output_size
                                : (size_t)outputArray.count;
    memcpy(output_data, outputPtr, outputCopySize * sizeof(float));

    return 0;
  }
}

CoreMLModelRef coreml_make_state(CoreMLModelRef model_ref) {
  @autoreleasepool {
    if (!model_ref) {
      NSLog(@"[CoreML] coreml_make_state: NULL model reference");
      return NULL;
    }

    MLModel *model = (__bridge MLModel *)model_ref;
    
    if (@available(macOS 14.0, *)) {
      NSError *error = nil;
      MLState *state = [model newState];
      if (!state) {
        NSLog(@"[CoreML] coreml_make_state: Failed to create state");
        return NULL;
      }
      NSLog(@"[CoreML] coreml_make_state: Created MLState successfully");
      return (__bridge_retained void *)state;
    } else {
      NSLog(@"[CoreML] coreml_make_state: MLState APIs require macOS 14.0+");
      return NULL;
    }
  }
}

void coreml_free_state(CoreMLModelRef state_ref) {
  if (state_ref) {
    CFRelease(state_ref);
  }
}

int coreml_predict_stateful(CoreMLModelRef model_ref, CoreMLModelRef state_ref,
                            const int32_t *input_ids,
                            size_t input_ids_size, const float *causal_mask,
                            size_t mask_size, float *output_data,
                            size_t output_size) {
  @autoreleasepool {
    if (!model_ref) {
      NSLog(@"[CoreML] Error: NULL model reference");
      return -1;
    }

    MLModel *model = (__bridge MLModel *)model_ref;
    NSError *error = nil;

    // Build inputIds as a MultiArray [1, L]
    NSArray<NSNumber *> *inputIdsShape = @[ @1, @(input_ids_size) ];
    MLMultiArray *inputIdsArray =
        [[MLMultiArray alloc] initWithShape:inputIdsShape
                                   dataType:MLMultiArrayDataTypeInt32
                                      error:&error];
    if (error || !inputIdsArray) {
      NSLog(@"[CoreML] Error creating inputIds array: %@",
            error.localizedDescription);
      return -2;
    }
    memcpy((int32_t *)inputIdsArray.dataPointer, input_ids,
           input_ids_size * sizeof(int32_t));

    // Build causalMask [1,1,L,L] if needed; your stateless model can ignore it
    // if absent.
    size_t mask_dim = (size_t)sqrt((double)mask_size);
    NSArray<NSNumber *> *maskShape = @[ @1, @1, @(mask_dim), @(mask_dim) ];
    MLMultiArray *causalMaskArray =
        [[MLMultiArray alloc] initWithShape:maskShape
                                   dataType:MLMultiArrayDataTypeFloat16
                                      error:&error];
    if (error || !causalMaskArray) {
      NSLog(@"[CoreML] Error creating causalMask array: %@",
            error.localizedDescription);
      return -3;
    }
    __fp16 *maskPtr = (__fp16 *)causalMaskArray.dataPointer;
    for (size_t i = 0; i < mask_size; i++) {
      maskPtr[i] = (__fp16)causal_mask[i];
    }

    NSMutableDictionary *inputDict =
        [@{@"inputIds" : inputIdsArray, @"causalMask" : causalMaskArray}
            mutableCopy];

    MLDictionaryFeatureProvider *inputProvider =
        [[MLDictionaryFeatureProvider alloc] initWithDictionary:inputDict
                                                          error:&error];
    if (error || !inputProvider) {
      NSLog(@"[CoreML] Error creating feature provider: %@",
            error.localizedDescription);
      return -4;
    }

    id<MLFeatureProvider> prediction = nil;
    
    if (@available(macOS 14.0, *)) {
      if (state_ref) {
        MLState *state = (__bridge MLState *)state_ref;
        prediction = [model predictionFromFeatures:inputProvider
                                        usingState:state
                                             error:&error];
      } else {
        prediction = [model predictionFromFeatures:inputProvider
                                             error:&error];
      }
    } else {
      prediction = [model predictionFromFeatures:inputProvider
                                           error:&error];
    }

    if (error || !prediction) {
      NSLog(@"[CoreML] Prediction error: %@", error.localizedDescription);
      return -5;
    }

    MLFeatureValue *outputFeature = [prediction featureValueForName:@"logits"];
    if (!outputFeature || outputFeature.type != MLFeatureTypeMultiArray) {
      NSLog(@"[CoreML] Error: Invalid logits output");
      return -6;
    }

    MLMultiArray *outputArray = outputFeature.multiArrayValue;
    size_t copySize = output_size < (size_t)outputArray.count
                          ? output_size
                          : (size_t)outputArray.count;

    if (outputArray.dataType == MLMultiArrayDataTypeFloat16) {
      __fp16 *src = (__fp16 *)outputArray.dataPointer;
      for (size_t i = 0; i < copySize; i++) {
        output_data[i] = (float)src[i];
      }
    } else if (outputArray.dataType == MLMultiArrayDataTypeFloat32) {
      float *src = (float *)outputArray.dataPointer;
      memcpy(output_data, src, copySize * sizeof(float));
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

int coreml_get_input_shape(CoreMLModelRef model_ref, const char *input_name,
                           int64_t *shape_out, size_t max_dims) {
  @autoreleasepool {
    if (!model_ref) {
      return -1;
    }

    MLModel *model = (__bridge MLModel *)model_ref;
    NSString *inputNameStr = [NSString stringWithUTF8String:input_name];

    MLModelDescription *description = model.modelDescription;
    MLFeatureDescription *inputDesc =
        description.inputDescriptionsByName[inputNameStr];

    if (!inputDesc) {
      return -2;
    }

    NSArray<NSNumber *> *shape = inputDesc.multiArrayConstraint.shape;
    size_t dims = shape.count < max_dims ? shape.count : max_dims;

    for (size_t i = 0; i < dims; i++) {
      shape_out[i] = [shape[i] longLongValue];
    }

    return (int)dims;
  }
}
