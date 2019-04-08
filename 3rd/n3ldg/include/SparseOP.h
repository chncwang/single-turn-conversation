/*
 * SparseOP.h
 *
 *  Created on: Jul 20, 2016
 *      Author: mason
 */

#ifndef SPARSEOP_H_
#define SPARSEOP_H_

#include "MyLib.h"
#include "Alphabet.h"
#include "Node.h"
#include "Graph.h"
#include "SparseParam.h"

// for sparse features
class SparseParams {
  public:
    SparseParam W;
    Alphabet elems;
    int nVSize;
    int nDim;

  public:
    SparseParams() {
        nVSize = 0;
        nDim = 0;
    }

    void exportAdaParams(ModelUpdate& ada) {
        ada.addParam(&W);
    }

    void initWeights(int nOSize) {
        if (nVSize == 0) {
            std::cerr << "nVSize is 0" << std::endl;
            abort();
        }
        nDim = nOSize;
        W.init(nOSize, nVSize);
    }


    void init(const Alphabet &alpha, int nOSize) {
        elems = alpha;
        nVSize = elems.size();
        initWeights(nOSize);
    }

    int getFeatureId(const string& strFeat) {
        int idx = elems.from_string(strFeat);
        return idx;
    }

};

//only implemented sparse linear node.
//non-linear transformations are not support,
class SparseNode : public Node {
  public:
    SparseParams* param;
    vector<int> ins;

    SparseNode() : Node() {
        ins.clear();
        param = NULL;
        node_type = "sparsenode";
    }

    void setParam(SparseParams* paramInit) {
        param = paramInit;
    }

    //notice the output
    void forward(Graph *cg, const vector<string>& x) {
        int featId;
        int featSize = x.size();
        for (int idx = 0; idx < featSize; idx++) {
            featId = param->getFeatureId(x[idx]);
            if (featId >= 0) {
                ins.push_back(featId);
            }
        }
        degree = 0;
        cg->addNode(this);
    }

    void compute() {
        param->W.value(ins, val);
    }

    //no output losses
    void backward() {
        //assert(param != NULL);
        param->W.loss(ins, loss);
    }

    PExecute generate();

    // better to rewrite for deep understanding
    bool typeEqual(PNode other) {
        bool result = Node::typeEqual(other);
        if (!result) return false;

        SparseNode* conv_other = (SparseNode*)other;
        if (param != conv_other->param) {
            return false;
        }

        return true;
    }

};

class SparseExecute :public Execute {};

PExecute SparseNode::generate() {
    SparseExecute* exec = new SparseExecute();
    exec->batch.push_back(this);
    return exec;
}

#endif /* SPARSEOP_H_ */
