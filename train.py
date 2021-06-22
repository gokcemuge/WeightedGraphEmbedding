from utilities.model_initalization import Model
from utilities.evaluation import *
from utilities.data_utilities import KnowledgeGraph
from utilities.data_utilities import get_minibatches, gen_psl_samples
from utilities.loss_functions import *
import torch
import numpy as np
import time
from operator import itemgetter
import itertools
import argparse


def train(name, out_file, model, data_dir, dim, batch_size,
          val_batch_size, test_batch_size, lr, reg, reg_scale, max_epoch,
          validate_epoch, test_epoch, negative_sample,
          rule_lam, train_with_groundings, train_with_psl, plot,
          ndcg_check, hr_map_count, save_model, load_model, margin,
          lambda_1, lambda_2):

    random_seed = 999
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    kg = KnowledgeGraph(data_dir=data_dir, train_with_groundings=train_with_groundings, train_with_psl=train_with_psl)
    model = model(name, kg=kg, embedding_dim=dim, batch_size=batch_size, learning_rate=lr,
                  neg_size=negative_sample* 2,
                  gpu=True,  train_with_groundings=train_with_groundings,
                  train_with_psl=train_with_psl, save_model=save_model, load_model=load_model)

    optimizer = torch.optim.Adam(model.parameters(), model.learning_rate)

    if model.name == 'UKGE_rect' or model.name == 'WGE_rect':
        reg = True
    else:
        reg = False

    print(". . . . ")
    print(torch.cuda.get_device_name(0))
    print("model: ", model.name)
    print("data: ", kg.data_dir)
    print("dimension: ", model.embedding_dim)
    print("batch size: ", model.batch_size)
    print("val batch size: ", val_batch_size)
    # print("test batch size: ", test_batch_size)
    print("lr: ", model.learning_rate)
    print("regularization: ", reg)
    if reg:
        print("reg coeff: ", reg_scale)
    print("hr_map_count: ", hr_map_count)
    print("train_with_groundings: ", train_with_groundings)
    print("train_with_psl: ", train_with_psl)

    print("negative sampling size: ", negative_sample * 2)
    print("rule coefficient: ", rule_lam)

    if model.name == 'transE' or model.name == 'complEx' or model.name == 'distmult':
        print("margin: ", margin)

    if model.name == 'WGE_logi' or model.name == 'WGE_rect':
        print("lambda_1: ", lambda_1)
        print("lambda_2: ", lambda_2)

    print("max epoch: ", max_epoch)
    print("validation every epoch: ", validate_epoch)
    print("test every epoch: ", test_epoch)
    print("plot mse: ", plot)

    print("load_model: ", model.load_model)
    print("save_model: ", model.save_model)

    strong_threshold = 0.85
    print("threshold for strong facts: ", strong_threshold)
    print(". . . . ")

    entity_list = list(kg.id2ent_dict.keys())  # kg.entity_dict  = id to entity

    train_pos = np.array(kg.training_triples)
    test_pos = np.array(kg.test_triples)
    val_pos = np.array(kg.validation_triples)

    if model.name == 'transE' or model.name == 'complEx' or model.name == 'distmult':
        # print("train, test, val data is filtered")

        train_pos_org = train_pos
        test_pos_org = test_pos
        val_pos_org = val_pos

        print("train data filtered for strong facts only")
        train_pos = train_pos[(train_pos[:, 3].astype(np.float) > strong_threshold)]

        print("weak test data for classification is created")
        test_neg = np.split(sample_negatives_only(test_pos, 1, entity_list, kg.all_pos_triples), 2)[1]

    else:

        print("weak test data for classification is created")
        test_neg = np.split(sample_negatives_only(test_pos, 1, entity_list, kg.all_pos_triples), 2)[1]

    if train_with_psl:
        psl_loss = 0
        psl_triples_all = np.array(kg.psl_triples)
        psl_batch_size_new = len(psl_triples_all) // (kg.n_training_triple // batch_size)+ 1



    if train_with_groundings:
        rule_loss = 0
        groundings_all = np.array(kg.groundings)
    else:
        groundings_all = np.array([0,1,2] ) # dummy init, just for loop operation

    # for evaluation
    #hr_map200 = kg.get_fixed_hr(200)
    #hr_map200 = kg.get_fixed_hr(hr_map_count)

    if load_model:
        # load model
        model.load_embedding()
        print('#Model is loaded!')

        val_start_time = time.time()
        model.eval()

        if model.name == "transE" or model.name == 'distmult' or model.name == 'complEx':
            with torch.no_grad():
                decision_tree_classify(model, strong_threshold, train_pos_org, test_pos_org, test_neg)

                print('..................................')
                return

        with torch.no_grad():
            print('Evaluating on Test Dataset...')
            decision_tree_classify(model, strong_threshold, train_pos, test_pos, test_neg)

            mse_pos, mae_pos = get_mse_pos(model, test_pos, val_batch_size, plot)
            mse_neg, mae_neg = get_mse_neg(model, test_pos, val_batch_size, negative_sample,
                                           entity_list, kg.all_pos_triples, plot)
            val_duration = time.time() - val_start_time
            print('mse tot: {:.4f} - mae tot: {:.4f} - test time: {:.1f}'.format(mse_pos.item() + mse_neg.item(),
                                                                                 mae_pos.item() + mae_neg.item(),
                                                                                 val_duration))
        return


    print('#Training Started!')
    training_start_time = time.time()
    for epoch in range(max_epoch):
        # losses_epoch = []  # actually this is just for history
        epoch_start_time = time.time()
        model.train()

        train_triple = list(get_minibatches(train_pos, batch_size, shuffle=True))

        # one grounded_triple [h,t,r,w]
        grounding_batch_size = (len(groundings_all) // (kg.n_training_triple // batch_size))+1
        grounded_triples = list(get_minibatches(groundings_all, grounding_batch_size, shuffle=True))

        single_epoch_loss = []
        single_epoch_loss_pos = []
        single_epoch_loss_neg = []

        epoch_total_loss = 0

        for iter_triple, grounding_batch in itertools.zip_longest(train_triple, grounded_triples):  # h t r
            try:
                iter_triple_size = iter_triple.shape[0]
            except:
                break
            iter_triple_size = iter_triple.shape[0]

            iter_neg = sample_negatives_only(iter_triple, negative_sample, entity_list, kg.all_pos_but_test_triples)
            pos_weights = iter_triple[:, 3].astype(np.float64)

            if model.name == 'rescal':
                pos_score = model.forward(iter_triple)  # same as f_prob_h
                neg_score = model.forward(iter_neg)  # same as (f_prob_hn , f_prob_tn)
                pos_loss, negative_loss = UKGE_main_loss(model, pos_score, pos_weights, neg_score)
                loss = pos_loss + negative_loss

            if model.name == "UKGE_logi":

                pos_score = model.forward(iter_triple)  # same as f_prob_h
                neg_score = model.forward(iter_neg)  # same as (f_prob_hn , f_prob_tn)
                pos_loss, negative_loss = UKGE_main_loss(model, pos_score, pos_weights, neg_score)
                loss = pos_loss + negative_loss

                if model.train_with_psl:
                    # one psl_triple [h,t,r,w]
                    psl_batch_size = psl_batch_size_new
                    psl_triples_batch = gen_psl_samples(psl_triples_all, psl_batch_size)
                    psl_weights = psl_triples_batch[:, 3].astype(np.float64)

                    psl_scores = model.forward(psl_triples_batch)
                    psl_loss = UKGE_psl_loss(model, psl_weights, psl_scores)

                    loss = pos_loss + negative_loss + psl_loss

                if model.train_with_groundings:
                    try:
                        grounding_weights = grounding_batch[:, 3].astype(np.float64)
                        grounding_score = model.forward(grounding_batch)
                        rule_loss = WGE_rule_loss(model, grounding_weights, grounding_score)

                        loss = pos_loss + negative_loss + (rule_loss * rule_lam)
                    except:
                        break

            if model.name == "UKGE_rect":
                model.regul = True
                pos_score, regularizer = model.forward(iter_triple)  # same as f_prob_h
                model.regul = False

                neg_score = model.forward(iter_neg)  # same as (f_prob_hn , f_prob_tn)
                pos_loss, negative_loss = UKGE_main_loss(model, pos_score, pos_weights, neg_score)
                loss = pos_loss + negative_loss + regularizer*reg_scale

                if model.train_with_psl:
                    # one psl_triple [h,t,r,w]
                    psl_batch_size = psl_batch_size_new
                    psl_triples_batch = gen_psl_samples(psl_triples_all, psl_batch_size)
                    psl_weights = psl_triples_batch[:, 3].astype(np.float64)

                    psl_scores = model.forward(psl_triples_batch)
                    psl_loss = UKGE_psl_loss(model, psl_weights, psl_scores)
                    loss = pos_loss + negative_loss + regularizer*reg_scale + psl_loss

                if model.train_with_groundings:
                    try:
                        grounding_weights = grounding_batch[:, 3].astype(np.float64)
                        grounding_score = model.forward(grounding_batch)
                        grounding_score = torch.clamp(torch.clamp(grounding_score, max=1), min=0)

                        rule_loss = WGE_rule_loss(model, grounding_weights, grounding_score)

                        loss = pos_loss + negative_loss + (regularizer * reg_scale) + (rule_loss * rule_lam)
                    except:
                        break

            if model.name == "WGE_logi" or model.name == "WGE_logi_rescal":

                keys = list(zip(iter_triple[:, 0], iter_triple[:, 1], iter_triple[:, 2]))
                triple_idxs = list(itemgetter(*keys)(kg.triple2idx))  # select triples to be used in loss

                epsilons_left_batch = model.epsilons_left(torch.tensor(triple_idxs).cuda())
                epsilons_right_batch = model.epsilons_right(torch.tensor(triple_idxs).cuda())

                pos_score = model.forward(iter_triple)
                neg_score = model.forward(iter_neg)  # when the model rect, bound is needed after forward

                pos_loss, negative_loss = WGE_loss(model, pos_score, pos_weights, neg_score,
                                                   epsilons_left_batch, epsilons_right_batch, lambda_1, lambda_2)

                loss = pos_loss + negative_loss

                if model.train_with_groundings:
                    try:
                        grounding_weights = grounding_batch[:, 3].astype(np.float64)
                        grounding_score = model.forward(grounding_batch)
                        rule_loss = WGE_rule_loss(model, grounding_weights, grounding_score)

                        loss = pos_loss + negative_loss + (rule_loss * rule_lam)
                    except:
                        break

            if model.name == "WGE_rect" or model.name == "WGE_rect_rescal":

                keys = list(zip(iter_triple[:, 0], iter_triple[:, 1], iter_triple[:, 2]))
                triple_idxs = list(itemgetter(*keys)(kg.triple2idx))  # select triples to be used in loss

                epsilons_left_batch = model.epsilons_left(torch.tensor(triple_idxs).cuda())
                epsilons_right_batch = model.epsilons_right(torch.tensor(triple_idxs).cuda())

                model.regul = True
                pos_score, regularizer = model.forward(iter_triple)
                model.regul = False

                neg_score = model.forward(iter_neg)

                pos_loss, negative_loss = WGE_loss(model, pos_score, pos_weights, neg_score,
                                                   epsilons_left_batch, epsilons_right_batch, lambda_1, lambda_2)

                loss = pos_loss + negative_loss + (regularizer * reg_scale)

                if model.train_with_groundings:
                    try:
                        grounding_weights = grounding_batch[:, 3].astype(np.float64)
                        grounding_score = model.forward(grounding_batch)
                        # clamp is based on UKGE implementation
                        grounding_score = torch.clamp(torch.clamp(grounding_score, max=1), min=0)

                        rule_loss = WGE_rule_loss(model, grounding_weights, grounding_score)

                        loss = pos_loss + negative_loss + (regularizer * reg_scale) + (rule_loss * rule_lam)
                    except:
                        break

            if model.name == "transE" or model.name == 'distmult' or model.name == 'complEx':
                pos_score = model.forward(iter_triple)  # same as f_prob_h
                neg_score = model.forward(iter_neg)  # same as (f_prob_hn , f_prob_tn)

                loss, pos_score, neg_score = adversarial_loss(model, pos_score, neg_score, margin=margin,
                                                              negative_sample_number=5, temperature=0)

                pos_loss = pos_score  # dummy
                negative_loss = neg_score  # dummy

            total_loss = loss

            single_epoch_loss.append(total_loss.item())
            single_epoch_loss_pos.append(pos_loss.item())
            single_epoch_loss_neg.append(negative_loss.item())
            epoch_total_loss = epoch_total_loss + total_loss

            # batch gradient descent
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if iter_triple_size < batch_size or iter_triple_size == 1:
                '''Last Batch'''
                break
            '''............end of mini-batch iterations of one epoch..............'''

        total_loss_average = sum(single_epoch_loss) / len(single_epoch_loss)
        pos_loss_print = sum(single_epoch_loss_pos) / len(single_epoch_loss)
        neg_loss_print = sum(single_epoch_loss_neg) / len(single_epoch_loss)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print('Epoch-{}: Total loss: {:.4f} - Pos loss: {:.4f} - Neg loss: {:.8f} - Epoch time {:.1f}' .format(
            epoch + 1,
            total_loss_average,
            pos_loss_print,
            neg_loss_print,
            epoch_duration))

        if model.name == "transE" or model.name == 'distmult' or model.name == 'complEx':
            if (epoch + 1) % validate_epoch == 0:
                val_start_time = time.time()
                model.eval()
                with torch.no_grad():
                    decision_tree_classify(model, strong_threshold, train_pos_org, test_pos_org, test_neg)

                    if model.save_model:
                        model.save_embedding(epoch+1)

                    print('..................................')
        else:
            if (epoch+1) % validate_epoch == 0:
                val_start_time = time.time()
                model.eval()
                with torch.no_grad():
                    print('Evaluating on Validation Dataset...')

                    mse_pos, mae_pos = get_mse_pos(model, val_pos, val_batch_size, plot)
                    mse_neg, mae_neg = get_mse_neg(model, val_pos, val_batch_size, negative_sample, entity_list,
                                                   kg.all_pos_but_test_triples, plot)

                    val_duration = time.time() - val_start_time
                    print('mse tot: {:.4f} - mae tot: {:.4f} - val time: {:.1f}'.format(mse_pos+mse_neg,
                                                                          mae_pos+mae_neg, val_duration))

                    if model.save_model:
                        model.save_embedding(epoch+1)

                    print('..................................')

            if (epoch + 1) % test_epoch == 0:
                val_start_time = time.time()
                model.eval()
                with torch.no_grad():

                    print('Evaluating on Test Dataset...')
                    decision_tree_classify(model, strong_threshold, train_pos, test_pos, test_neg)

                    mse_pos, mae_pos = get_mse_pos(model, test_pos, val_batch_size, plot)
                    mse_neg, mae_neg = get_mse_neg(model, test_pos, val_batch_size, negative_sample,
                                                   entity_list, kg.all_pos_triples, plot)
                    val_duration = time.time() - val_start_time
                    print('mse tot: {:.4f} - mae tot: {:.4f} - test time: {:.1f}'.format(mse_pos + mse_neg,
                                                                                        mae_pos + mae_neg,
                                                                                        val_duration))




def main():
        print('running')
        train(out_file=out_file,
              name=model_name,
              model=Model,
              data_dir=data_dir,
              batch_size=batch_size,
              val_batch_size=val_batch_size,
              test_batch_size=test_batch_size,
              dim=dim,
              lr=lr,
              reg=reg,
              reg_scale=reg_scale,
              negative_sample=negative_sample,
              rule_lam=rule_lam,
              max_epoch=max_epoch,
              validate_epoch=validate_epoch,
              test_epoch=test_epoch,
              train_with_groundings=train_with_groundings,
              train_with_psl=train_with_psl,
              plot=plot,
              ndcg_check=ndcg_check,
              hr_map_count=hr_map_count,
              save_model=save_model,
              load_model=load_model,
              margin=margin,
              lambda_1=lambda_1,
              lambda_2=lambda_2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda', action='store_false', help='use GPU')

    # parser.add_argument('--do_train', action='store_true')
    # parser.add_argument('--do_valid', action='store_true')
    # parser.add_argument('--do_test', action='store_true')
    # parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')
    parser.add_argument('-out_file', "--output_file_name", default='training_results', help="Output file name", type=str)

    parser.add_argument('-m', "--model_name", default='WGE_logi', help="Model name", type=str)
    parser.add_argument('-data', "--dataset", default='aida35k', help="Dataset name",  type=str)
    parser.add_argument('-neg', '--negative_sample', default=10, type=int)
    parser.add_argument('-lr', '--learning_rate', default=0.001, help="learning rate", type=float)
    parser.add_argument("-dim", "--dimension", default=128, help="dimension size of embeddings", type=int)
    parser.add_argument('-b', '--batch_size', default=512, type=int)
    parser.add_argument('-val_b', '--validation_batch_size', default=512, type=int)
    parser.add_argument('-test_b', '--test_batch_size', default=512, type=int)

    parser.add_argument('-max', '--max_steps', default=100, type=int, help="maximum number of epochs")
    parser.add_argument('-val_e', '--validate_epoch', default=10, type=int, help="validate every x epoch")
    parser.add_argument('-test_e', '--test_epoch', default=100, type=int, help="test every x epoch")

    parser.add_argument('-reg', '--regul', default="false", type=str, help='Use Regularization or not')
    parser.add_argument('-reg_scale', '--reg_scale', default=0.0005, help="reg scale of L2", type=float)
    parser.add_argument('-rule_coef', '--rule_lam', default=0.05, help='Regularization coefficient of Rule loss', type=float)

    parser.add_argument('-margin', '--margin', default=4.0, help="margin of the adverserial loss", type=float)

    parser.add_argument('-lambda_1', '--lambda_1', default=1.0, help="lambda of left parameters", type=float)
    parser.add_argument('-lambda_2', '--lambda_2', default=1.0, help="lambda of right parameters", type=float)

    parser.add_argument('-groundings', '--train_with_groundings', default="false", type=str)
    parser.add_argument('-psl', '--train_with_psl', default="false", type=str)
    parser.add_argument('-plot', '--plot_mse', default="false", type=str)
    parser.add_argument('-ndcg', default="false", type=str)

    parser.add_argument('-hr_map', '--hr_map_count', default=20, help="hr_map_queries", type=int)

    parser.add_argument('-load', '--load_model', default="false", type=str)
    parser.add_argument('-save', '--save_model', default="false", type=str)

    args = parser.parse_args()
    out_file = args.output_file_name
    model_name = args.model_name
    data_dir = "dataset/" + str(args.dataset)
    batch_size = args.batch_size
    val_batch_size = args.validation_batch_size
    test_batch_size = args.test_batch_size
    dim = args.dimension
    lr = args.learning_rate
    reg_scale = args.reg_scale
    max_epoch = args.max_steps
    validate_epoch = args.validate_epoch
    test_epoch = args.test_epoch
    ndcg_check = args.ndcg
    margin = args.margin

    lambda_1 = args.lambda_1
    lambda_2 = args.lambda_2

    negative_sample = int((args.negative_sample / 2))  # half for head, half for tail will be used
    rule_lam = args.rule_lam

    hr_map_count = args.hr_map_count

    if str(args.regul).lower() == "true":
        reg = True
    elif str(args.regul).lower() == "false":
        reg = False

    if str(args.train_with_groundings).lower() == "true":
        train_with_groundings = True
    elif str(args.train_with_groundings).lower() == "false":
        train_with_groundings = False

    if str(args.train_with_psl).lower() == "true":
        train_with_psl = True
    elif str(args.train_with_psl).lower() == "false":
        train_with_psl = False

    if str(args.plot_mse).lower() == "true":
        plot = True
    elif str(args.plot_mse).lower() == "false":
        plot = False

    if str(args.ndcg).lower() == "true":
        ndcg_check = True
    elif str(args.ndcg).lower() == "false":
        ndcg_check = False

    if str(args.load_model).lower() == "true":
        load_model = True
    elif str(args.load_model).lower() == "false":
        load_model = False

    if str(args.save_model).lower() == "true":
        save_model = True
    elif str(args.save_model).lower() == "false":
        save_model = False

    '''
    model_name = "rescal"
    # model_name = "UKGE_logi"
    plot = False
    load_model = False
    data_dir = "dataset/" + "aida35k"
    lr = 0.001
    validate_epoch = 10
    test_epoch = 50
    max_epoch = 50
    batch_size = 128
    dim = 128
    reg_scale = 0.01
    negative_sample = 5  # this will be used for head&tail so it is actually this value X 2
    ndcg_check = True
    train_with_psl = False
    train_with_groundings = False
    '''

    main()
