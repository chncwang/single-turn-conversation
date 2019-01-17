#ifndef SINGLE_TURN_CONVERSATION_BLEU_H
#define SINGLE_TURN_CONVERSATION_BLEU_H

#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <boost/format.hpp>
#include "conversation_structure.h"

const int BLEU_MAX_N = 4;

struct CandidateAndReferences {
    std::vector<int> candidate;
    std::vector<std::vector<int>> references;

    CandidateAndReferences() = default;

    CandidateAndReferences(const std::vector<int> &c, const std::vector<std::vector<int>> &ref) {
        candidate = c;
        references = ref;
    }
};

float mostMatchedCount(const CandidateAndReferences &candidate_and_references,
        int gram_len) {
    int max_mached_count = 0;
    const auto &references = candidate_and_references.references;
    const auto &candidate = candidate_and_references.candidate;
    for (const std::vector<int> &reference : references) {
        int matched_count = 0;
        for (int i = 0; i < candidate.size() + 1 - gram_len; ++i) {
            std::vector<bool> matched;
            for (int j = 0; j < reference.size() + 1 - gram_len; ++j) {
                matched.push_back(false);
            }

            for (int j = 0; j < reference.size() + 1 - gram_len; ++j) {
                if (matched.at(j)) {
                    continue;
                }

                bool finded = false;
                for (int k = 0; k < gram_len; ++k) {
                    if (candidate.at(i + k) != reference.at(j + k)) {
                        break;
                    }
                    if (k == gram_len - 1) {
                        finded = true;
                    }
                }

                if (finded) {
                    matched.at(j) = true;
                    matched_count++;
                    break;
                }
            }
        }

        if (matched_count > max_mached_count) {
            max_mached_count = matched_count;
        }
    }

    return max_mached_count;
}

int mostMatchedLength(const CandidateAndReferences &candidate_and_references) {
    int candidate_len = candidate_and_references.candidate.size();
    auto cmp = [&](const std::vector<int> &a, const std::vector<int> &b)->bool {
        return abs(candidate_len - a.size()) < abs(candidate_len - b.size());
    };
    return std::min_element(candidate_and_references.references.begin(),
            candidate_and_references.references.end(), cmp)->size();
}

float computeBleu(const std::vector<CandidateAndReferences> &candidate_and_references_vector) {
    static const int MAX_GRAM_LEN = 4;
    float weighted_sum = 0.0f;
    int r_sum = 0;
    int c_sum = 0;

    for (int gram_len = 1; gram_len <= MAX_GRAM_LEN; ++gram_len) {
        int matched_count_sum = 0;
        int candidate_count_sum = 0;
        for (const auto &candidate_and_references : candidate_and_references_vector) {
            int matched_count = mostMatchedCount(candidate_and_references, gram_len);
            matched_count_sum += matched_count;
            candidate_count_sum += candidate_and_references.candidate.size() + 1 - gram_len;

            int r = mostMatchedLength(candidate_and_references);
            r_sum += r;
        }
        c_sum += candidate_count_sum;

        weighted_sum += 1.0f / MAX_GRAM_LEN * log(static_cast<float>(matched_count_sum) /
                candidate_count_sum);
    }

    float bp = c_sum > r_sum ? 1.0f : exp(1 - static_cast<float>(r_sum) / c_sum);
    return bp * exp(weighted_sum);
}

#endif
