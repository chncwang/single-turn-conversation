#ifndef SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_DECODER_COMPONENTS_BUILDER_H
#define SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_DECODER_COMPONENTS_BUILDER_H

#include <memory>
#include "single_turn_conversation/encoder_decoder/global_context/global_context_decoder_components.h"

std::shared_ptr<DecoderComponents> buildDecoderComponents() {
    return std::shared_ptr<GlobalContextDecoderComponents>(new GlobalContextDecoderComponents);
}

#endif
