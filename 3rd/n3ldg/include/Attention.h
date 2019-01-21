#ifndef ATTENTION_BUILDER
#define ATTENTION_BUILDER

/*
*  Attention.h:
*  a set of attention builders
*
*  Created on: Apr 22, 2017
*      Author: mszhang
*/

#include "MyLib.h"
#include "Node.h"
#include "BiOP.h"
#include "UniOP.h"
#include "Graph.h"
#include "AttentionHelp.h"
#include <memory>

struct AttentionParams {
    BiParams bi_atten;
    int hidden_dim;
    int guide_dim;

    AttentionParams() {
    }

    void exportAdaParams(ModelUpdate& ada) {
        bi_atten.exportAdaParams(ada);
    }

    void init(int nHidden, int nGuide) {
        bi_atten.init(1, nHidden, nGuide, false);
        hidden_dim = nHidden;
        guide_dim = nGuide;
    }
};

class AttentionBuilder {
public:
    vector<shared_ptr<BiNode>> _weights;
    AttentionSoftMaxNode _hidden;

    AttentionParams* _param;

    void init(AttentionParams &paramInit) {
        _param = &paramInit;
        _hidden.init(paramInit.hidden_dim);
    }

    void forward(Graph &cg, vector<Node *>& x, Node& guide) {
        if (x.size() == 0) {
            std::cerr << "empty inputs for lstm operation" << std::endl;
            abort();
        }

        if (x.at(0)->dim != _param->hidden_dim || guide.dim != _param->guide_dim) {
            std::cerr << "input dim does not match for attention  operation" << std::endl;
            abort();
        }

        for (int idx = 0; idx < x.size(); idx++) {
            shared_ptr<BiNode> weight_node(new BiNode);
            weight_node->setParam(_param->bi_atten);
            weight_node->init(1);
            weight_node->forward(cg, *x.at(idx), guide);
            _weights.push_back(weight_node);
        }
        vector<Node *> weights = toNodePointers<BiNode>(_weights);
        _hidden.forward(cg, x, weights);
    }
};



struct AttentionVParams {
    BiParams bi_atten;
    int hidden_dim;
    int guide_dim;

    AttentionVParams() {
    }

    void exportAdaParams(ModelUpdate& ada) {
        bi_atten.exportAdaParams(ada);
    }

    void init(int nHidden, int nGuide) {
        bi_atten.init(nHidden, nHidden, nGuide, false);
        hidden_dim = nHidden;
        guide_dim = nGuide;
    }
};

class AttentionVBuilder {
  public:
    int _nSize;
    int _nHiddenDim;
    int _nGuideDim;

    vector<BiNode> _weights;
    AttentionSoftMaxVNode _hidden;
    AttentionVParams* _param;

  public:
    AttentionVBuilder() {
        clear();
    }

    ~AttentionVBuilder() {
        clear();
    }

  public:
    void resize(int maxsize) {
        _weights.resize(maxsize);
        _hidden.setParam(maxsize);
    }

    void clear() {
        _weights.clear();
    }

  public:
    void init(AttentionVParams* paramInit) {
        _param = paramInit;
        _nHiddenDim = _param->hidden_dim;
        _nGuideDim = _param->guide_dim;

        int maxsize = _weights.size();
        for (int idx = 0; idx < maxsize; idx++) {
            _weights.at(idx).setParam(&_param->bi_atten);
            _weights.at(idx).init(_nHiddenDim);
        }
        _hidden.init(_nHiddenDim);
    }

  public:
    void forward(Graph *cg, const vector<Node *>& x, Node * guide) {
        if (x.size() == 0) {
            std::cout << "empty inputs for lstm operation" << std::endl;
            return;
        }
        _nSize = x.size();
        if (x.at(0)->dim != _nHiddenDim || guide->dim != _nGuideDim) {
            std::cout << "input dim does not match for attention  operation" << std::endl;
            return;
        }

        vector<Node *> aligns;
        for (int idx = 0; idx < _nSize; idx++) {
            _weights.at(idx).forward(cg, x.at(idx), guide);
            aligns.push_back(&_weights.at(idx));
        }
        _hidden.forward(cg, x, aligns);
    }
};


struct SelfAttentionParams {
    UniParams uni_atten;
    int hidden_dim;

    SelfAttentionParams() {
    }

    void exportAdaParams(ModelUpdate& ada) {
        uni_atten.exportAdaParams(ada);
    }

    void init(int nHidden) {
        uni_atten.init(1, nHidden, false);
        hidden_dim = nHidden;
    }
};

class SelfAttentionBuilder {
public:
    int _nSize;
    int _nHiddenDim;

    vector<UniNode> _weights;
    AttentionSoftMaxNode _hidden;

    SelfAttentionParams* _param;

    void init(SelfAttentionParams* paramInit) {
        _param = paramInit;
        _nHiddenDim = _param->hidden_dim;

        int maxsize = _weights.size();
        for (int idx = 0; idx < maxsize; idx++) {
            _weights.at(idx).setParam(&_param->uni_atten);
            _weights.at(idx).init(1);
        }
        _hidden.init(_nHiddenDim);
    }

    void forward(Graph &cg, vector<Node *>& x) {
        if (x.size() == 0) {
            std::cout << "empty inputs for lstm operation" << std::endl;
            return;
        }
        _nSize = x.size();
        if (x.at(0)->dim != _nHiddenDim) {
            std::cout << "input dim does not match for attention  operation" << std::endl;
            return;
        }

        vector<Node *> aligns;
        for (int idx = 0; idx < _nSize; idx++) {
            _weights.at(idx).forward(cg, *x.at(idx));
            aligns.push_back(&_weights.at(idx));
        }

        _hidden.forward(cg, x, aligns);
    }
};



struct SelfAttentionVParams {
    UniParams uni_atten;
    int hidden_dim;

    SelfAttentionVParams() {
    }

    void exportAdaParams(ModelUpdate& ada) {
        uni_atten.exportAdaParams(ada);
    }

    void init(int nHidden) {
        uni_atten.init(nHidden, nHidden, false);
        hidden_dim = nHidden;
    }
};

class SelfAttentionVBuilder {
  public:
    int _nSize;
    int _nHiddenDim;

    vector<UniNode> _weights;
    AttentionSoftMaxVNode _hidden;
    SelfAttentionVParams* _param;

  public:
    SelfAttentionVBuilder() {
        clear();
    }

    ~SelfAttentionVBuilder() {
        clear();
    }

  public:
    void resize(int maxsize) {
        _weights.resize(maxsize);
        _hidden.setParam(maxsize);
    }

    void clear() {
        _weights.clear();
    }

  public:
    void init(SelfAttentionVParams* paramInit) {
        _param = paramInit;
        _nHiddenDim = _param->hidden_dim;

        int maxsize = _weights.size();
        for (int idx = 0; idx < maxsize; idx++) {
            _weights.at(idx).setParam(&_param->uni_atten);
            _weights.at(idx).init(_nHiddenDim);
        }
        _hidden.init(_nHiddenDim);
    }

  public:
    void forward(Graph *cg, const vector<Node *>& x) {
        if (x.size() == 0) {
            std::cout << "empty inputs for lstm operation" << std::endl;
            return;
        }
        _nSize = x.size();
        if (x.at(0)->dim != _nHiddenDim) {
            std::cout << "input dim does not match for attention  operation" << std::endl;
            return;
        }

        vector<Node *> aligns;
        for (int idx = 0; idx < _nSize; idx++) {
            _weights.at(idx).forward(cg, x.at(idx));
            aligns.push_back(&_weights.at(idx));
        }
        _hidden.forward(cg, x, aligns);
    }
};

#endif
