from __future__ import print_function, division
from data_processing import *
from ddpg_at import *
from closed_loop import *
import matplotlib.pyplot as plt
import argparse

#
parser = argparse.ArgumentParser()
#The hyper-parameter forClosed-loop
parser.add_argument('--mode', default='closed_train', type=str)  # mode = 'closed_train' or 'ddpg_train'
parser.add_argument('--closed_test', default=True, type=bool)
parser.add_argument('--ddpg_test', default=True, type=bool)
parser.add_argument('--window_size', default=129, type=int)
parser.add_argument('--batch_size', default=64, type=int)  # mini batch size
parser.add_argument('--closed_learning_rate', default=2*1e-4, type=int)  # mini batch size
parser.add_argument('--epoch', default=1, type=int)

#The hyper-parameter only for DDPG-AT model
parser.add_argument('--tau', default=0.005, type=float)  # target smoothing coefficient
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--test_iteration', default=10, type=int)
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--gamma', default=0.99, type=int)  # discounted factor
parser.add_argument('--capacity', default=1000000, type=int)  # replay buffer size 1000000
parser.add_argument('--seed', default=True, type=bool)
parser.add_argument('--random_seed', default=9527, type=int)
parser.add_argument('--sample_frequency', default=2000, type=int)
parser.add_argument('--render', default=False, type=bool)  # show UI or not
parser.add_argument('--log_interval', default=1000, type=int)  #
parser.add_argument('--load', default=False, type=bool)  # load model
parser.add_argument('--render_interval', default=100, type=int)  # after render_interval, the env.render() will work
parser.add_argument('--exploration_noise', default=0.1, type=float)
parser.add_argument('--max_episode', default=1000000, type=int)  # num of games,1000000
parser.add_argument("--learning-starts", type=int, default=25e3,
                    help="timestep to start learning")
parser.add_argument('--print_log', default=5, type=int)
parser.add_argument('--update_iteration', default=100, type=int)
parser.add_argument("--policy-frequency", type=int, default=2,
                    help="the frequency of training policy (delayed)")
parser.add_argument('--model', default=96, type=int)
parser.add_argument('--layer', default=6, type=int)
parser.add_argument('--head', default=1, type=int)
parser.add_argument('--action_dim', default=1, type=int)

parser.add_argument('--d_ff', default=512, type=int)
args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
script_name = os.path.basename(__file__)

state_dim=args.window_size
action_dim = args.action_dim
d_model = args.model
n_layers = args.layer
d_k = d_v = int(d_model / args.head)

if args.seed:
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)






if __name__ == '__main__':
    epochs = args.epoch
    train_start, train_end = "2013-6-20", "2013-6-23" # Select training and testing date
    test_start, test_end = "2013-7-06", "2013-7-07"
    building = 2

    window_size = args.window_size
    batch_size = args.batch_size
    meter_key1 = 'washing machine'
    meter_keys = [ 'washing machine', 'microwave', 'dish washer', 'fridge','kettle']  # select loads

    if args.mode=='closed_train':
        # data processing and return the normalized data
        learning_rate=args.closed_learning_rate
        train_main, train_appliances, test_main, test_appliances, train_meter_means, train_meter_stds = load_data(train_start, train_end, test_start,
                                                                             test_end, building, meter_keys,
                                                                             window_size, batch_size,method='closed',sample_period=6)
        # begin training

        closeddisaggregator=closedDisaggregator(len(meter_keys))
        closeddisaggregator.train_chunk(train_main,train_appliances, epochs,learning_rate,batch_size)
        if args.closed_test == True:  # if true begin testing
            results=closeddisaggregator.disaggregate(test_main,train_appliances)
            processed_data =  closeddisaggregator.postprocess(results,train_meter_means, train_meter_stds) #inverse the predictions
            file_path = closeddisaggregator.save_results(processed_data)#store the predictions,return a .csv file.
            print(f"Results saved to {file_path}")




    elif args.mode=='ddpg_train':
        # data processing and return the normalized data
        #mains, appliance1,appliance1_mean,appliance1_std =dataset(train_start,train_end,test_start,test_end,building,key1,key2,key3,key4,key5,window_size,batch_size,method='ddpg_train')
        train_main, train_appliances, test_main, test_appliances =load_data(train_start,train_end,test_start,test_end,building,meter_keys,window_size,batch_size,method='ddpg_train',sample_period=6)
        #test_nor_mains, test_nor_appliance1,appliance1_mean,appliance1_std =dataset(train_start,train_end,test_start,test_end,building,key1,key2,key3,key4,key5,window_size,batch_size,method='ddpg_test')


        state_dim = window_size
        action_dim = 1

        # establish the Enviroment

        env = Enviroment(train_main, train_appliances[0])

        # directory is the path of running log
        directory = './exp' + script_name + args.mode + "head" + str(args.head) + "model" + str(
            args.model) + "layer" + str(
            args.layer) + './'

        agent = DDPG(args.model, args.window_size,args.layer,action_dim,  directory,batch_size,args.tau,args.gamma,
                     args.policy_frequency,args.log_interval,d_k,d_v,args.head,args.d_ff,args.capacity)

        # whether load the trained model
        if args.load:
                agent.load()


        total_step = 0


        #Training of the ddpg model
        state = env.reset()
        for i in range(args.max_episode):
            total_reward = 0
            step = 0

            # The number of samples in the empirical pool is less than the learning_starts, continue to replenish it
            if i < args.learning_starts:
                action = agent.select_action(state)
            else:
                action = agent.select_action(state)

            next_state, reward, Ty, done = env.step(action)
            agent.replay_buffer.push((state, next_state, action, reward, float(done), Ty))

            state = next_state
            step += 1
            total_reward += reward
            print(
                "args.max_episode=%d i=%d Total Reward=%f................................" % (
                    args.max_episode, i, total_reward))

            # The number of samples in the empirical pool satisfy the threshold of learning_starts, begin to train
            if i > args.learning_starts:
                agent.update(i)
            # Saving models at log_interval
            if i % args.log_interval == 0:
                agent.save()

        # Testing of the ddpg model
        if args.ddpg_test == True:

            agent.load()
            st = time.time()
            train_dataset = TensorDataset(torch.from_numpy(test_main),
                                          torch.from_numpy(test_appliances).float())
            train_loader = tud.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            for i, (batch_test_mains, batch_test_appliance) in enumerate(train_loader):
                with torch.no_grad():

                    batch_mains_cuda = batch_test_mains.cuda()

                    if batch_mains_cuda.shape[0] == batch_size:

                        batch_pred1 = agent.actor(batch_mains_cuda)


                    else:

                        a = torch.zeros(batch_size - batch_mains_cuda.shape[0], 1, batch_mains_cuda.shape[2]).cuda()
                        b = torch.cat((batch_mains_cuda, a), 0).cuda()
                        batch_pred1 = agent.actor(b.view(-1, 1, state_dim))

                        batch_pred1 = batch_pred1[:batch_mains_cuda.shape[0], :]

                    if i == 0:
                        res1 = batch_pred1



                    else:
                        print('test_batch main', batch_mains_cuda)
                        print('test_batch pred2', batch_pred1)
                        res1 = torch.cat((res1, batch_pred1), dim=0)
                    i += 1

            ed = time.time()
            print("Inference Time consumption: {}s.".format(ed - st))

            appliance1_power = res1.to('cpu').detach().numpy()

            l2 = 1
            n = len(appliance1_power) + l2 - 1

            #Inverse normalisation of predicted values
            appliance1_power = appliance1_power * test_appliances.std() + test_appliances.mean()

            test_appliance1 = test_appliances[0] * test_appliances.std() + test_appliances.mean()

            valid_predictions1 = appliance1_power.flatten()

            appliance1_power = np.where(valid_predictions1 > 0, valid_predictions1, 0)
            # plot the truth and predicted values
            # plt.plot(appliance1_power)  # ，linewidth=5
            # plt.plot(test_appliance1)
            # plt.show()


            # save the truth and predicted values
            df1 = pd.DataFrame(appliance1_power)

            df2 = pd.DataFrame(test_appliance1)

            df_combined = pd.concat([df1, df2], axis=1)


            df_combined.to_csv(directory + 'combined_data.csv', index=False)

            # calculate the RAE
            sum_samples = 0.0
            sum_samples += len(appliance1_power)
            E_pred = sum(appliance1_power)
            E_ground = sum(test_appliance1)

            RAE = abs(E_pred - E_ground) / float(max(E_pred, E_ground))
            print('RAE', RAE)



else:
        raise NameError("mode wrong!!!")
