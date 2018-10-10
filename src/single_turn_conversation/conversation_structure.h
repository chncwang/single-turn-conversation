#ifndef SINGLE_TURN_CONVERSATION_SRC_BASIC_CONVERSATION_STRUCTURE_H
#define SINGLE_TURN_CONVERSATION_SRC_BASIC_CONVERSATION_STRUCTURE_H

#include <vector>

struct PostAndResponses {
    int post_id;
    std::vector<int> response_ids;
};

struct ConversationPair {
    ConversationPair() = default;

    ConversationPair(int postid, int responseid) : post_id(postid),
    response_id(responseid) {}

    int post_id;
    int response_id;
};

#endif
