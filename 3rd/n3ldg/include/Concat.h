#ifndef CONCAT
#define CONCAT

/*
*  Concat.h:
*  concatenatation operation.
*
*  Created on: Apr 22, 2017
*      Author: mszhang
*/


#include "MyLib.h"
#include "Node.h"
#include "Graph.h"
#if USE_GPU
#include "N3LDG_cuda.h"
#endif
#include "profiler.h"

class ConcatNode : public Node {
public:
    vector<int> inDims;
    vector<PNode> ins;

    ConcatNode() : Node("concat") {
        inDims.clear();
        ins.clear();
    }

    void forward(Graph &cg, const vector<PNode>& x) {
        if (x.size() == 0) {
            std::cout << "empty inputs for concat" << std::endl;
            abort();
        }

        ins.clear();
        for (int i = 0; i < x.size(); i++) {
            ins.push_back(x[i]);
        }

        int nSize = ins.size();
        for (int i = 0; i < nSize; ++i) {
            ins[i]->addParent(this);
        }
        inDims.clear();
        int curDim = 0;
        for (int i = 0; i < nSize; ++i) {
            inDims.push_back(ins[i]->val().dim);
            curDim += inDims[i];
        }
        if (curDim != getDim()) {
            std::cerr << "input dim size not match" << curDim << "\t" << getDim() << std::endl;
            abort();
        }
        cg.addNode(this);
    }

    PExecute generate();

    // better to rewrite for deep understanding
    bool typeEqual(PNode other) {
        if (!Node::typeEqual(other)) {
            return false;
        }
        ConcatNode *o = static_cast<ConcatNode*>(other);
        if (inDims.size() != o->inDims.size()) {
            return false;
        }
        for (int i = 0; i < inDims.size(); ++i) {
            if (inDims.at(i) != o->inDims.at(i)) {
                return false;
            }
        }
        return true;
    }

    size_t typeHashCode() const override {
        size_t hash_code = Node::typeHashCode() ^
            std::hash<int>{}(inDims.size());
        int i = 0;
        for (int dim : inDims) {
            hash_code ^= (dim << (i++ % 16));
        }
        return hash_code;
    }

    void compute() {
        int nSize = ins.size();
        int offset = 0;
        for (int i = 0; i < nSize; ++i) {
            memcpy(val().v + offset, ins.at(i)->val().v,
                    inDims.at(i) * sizeof(dtype));
            offset += inDims[i];
        }
    }


    void backward() {
        int nSize = ins.size();
        int offset = 0;
        for (int i = 0; i < nSize; ++i) {
            for (int idx = 0; idx < inDims[i]; idx++) {
                ins[i]->loss()[idx] += loss()[offset + idx];
            }
            offset += inDims[i];
        }
    }

};

#if USE_GPU
class ConcatExecute : public Execute {
  public:
    int outDim;
    int inCount;

    void  forward() {
        int count = batch.size();

        std::vector<dtype*> in_vals, vals;
        in_vals.reserve(inCount * count);
        vals.reserve(count);
        for (Node *node : batch) {
            ConcatNode *concat = static_cast<ConcatNode*>(node);
            for (Node *in : concat->ins) {
                in_vals.push_back(in->val.value);
            }
            vals.push_back(node->val.value);
        }

        n3ldg_cuda::ConcatForward(in_vals, static_cast<ConcatNode*>(batch.at(0))->inDims, vals,
                count, inCount, outDim);
#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->compute();
            n3ldg_cuda::Assert(batch[idx]->val.verify("concat forward"));
        }
#endif
    }

    void backward() {
        int count = batch.size();
        std::vector<dtype*> in_losses, losses;
        in_losses.reserve(inCount * count);
        losses.reserve(count);
        for (Node *node : batch) {
            ConcatNode *concat = static_cast<ConcatNode*>(node);
            for (Node *in : concat->ins) {
                in_losses.push_back(in->loss.value);
            }
            losses.push_back(node->loss.value);
        }

        n3ldg_cuda::ConcatBackward(in_losses, static_cast<ConcatNode*>(batch.at(0))->inDims,
                losses, count, inCount, outDim);
#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->backward();
        }
        for (int idx = 0; idx < count; idx++) {
            for (int j = 0; j < inCount; ++j) {
                n3ldg_cuda::Assert(static_cast<ConcatNode *>(batch[idx])->
                        ins[j]->loss.verify("concat backward"));
            }
        }
#endif
    }
};
#else
class ConcatExecute : public Execute {
};
#endif

PExecute ConcatNode::generate() {
    ConcatExecute* exec = new ConcatExecute();
    exec->batch.push_back(this);
#if USE_GPU
    exec->inCount = this->ins.size();
    exec->outDim = 0;
    for (int d : inDims) {
        exec->outDim += d;
    }
#endif
    return exec;
}

#endif
