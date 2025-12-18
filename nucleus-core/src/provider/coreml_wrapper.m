#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>

typedef void* CoreMLModelRef;

CoreMLModelRef coreml_load_model(const char* model_path) {
    @autoreleasepool {
        NSString *path = [NSString stringWithUTF8String:model_path];
        NSURL *modelURL = [NSURL fileURLWithPath:path];
        
        NSError *error = nil;
        MLModel *model = [MLModel modelWithContentsOfURL:modelURL error:&error];
        
        if (error) {
            NSLog(@"Error loading CoreML model: %@", error.localizedDescription);
            return NULL;
        }
        
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
        if (!model_ref) {
            return -1;
        }
        
        MLModel *model = (__bridge MLModel*)model_ref;
        
        NSError *error = nil;
        MLModelDescription *description = model.modelDescription;
        
        NSString *inputNameStr = [NSString stringWithUTF8String:input_name];
        MLFeatureDescription *inputDesc = description.inputDescriptionsByName[inputNameStr];
        
        if (!inputDesc) {
            NSLog(@"Input feature '%@' not found in model", inputNameStr);
            return -2;
        }
        
        NSArray<NSNumber *> *shape = inputDesc.multiArrayConstraint.shape;
        MLMultiArray *inputArray = [[MLMultiArray alloc] initWithShape:shape
                                                               dataType:MLMultiArrayDataTypeFloat32
                                                                  error:&error];
        if (error) {
            NSLog(@"Error creating input array: %@", error.localizedDescription);
            return -3;
        }
        
        float *dataPtr = (float *)inputArray.dataPointer;
        size_t copySize = MIN(input_size, inputArray.count);
        memcpy(dataPtr, input_data, copySize * sizeof(float));
        
        MLDictionaryFeatureProvider *inputProvider = [[MLDictionaryFeatureProvider alloc]
            initWithDictionary:@{inputNameStr: inputArray}
            error:&error];
        
        if (error) {
            NSLog(@"Error creating feature provider: %@", error.localizedDescription);
            return -4;
        }
        
        id<MLFeatureProvider> prediction = [model predictionFromFeatures:inputProvider error:&error];
        
        if (error) {
            NSLog(@"Prediction error: %@", error.localizedDescription);
            return -5;
        }
        
        NSString *outputNameStr = [NSString stringWithUTF8String:output_name];
        MLFeatureValue *outputFeature = [prediction featureValueForName:outputNameStr];
        
        if (!outputFeature || outputFeature.type != MLFeatureTypeMultiArray) {
            NSLog(@"Output feature '%@' not found or invalid type", outputNameStr);
            return -6;
        }
        
        MLMultiArray *outputArray = outputFeature.multiArrayValue;
        float *outputPtr = (float *)outputArray.dataPointer;
        size_t outputCopySize = MIN(output_size, outputArray.count);
        memcpy(output_data, outputPtr, outputCopySize * sizeof(float));
        
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
