#ifndef SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_MODEL_PARAMS_H
#define SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_MODEL_PARAMS_H

#include "N3LDG.h"

struct ModelParams {
    LookupTable lookup_table;
    LSTM1Params encoder_params;
};

#endif
