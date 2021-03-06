import numpy as np
import matplotlib,sys
matplotlib.use('agg')
import torch
from torch import nn
import utils
from BatchIterator import PaddedDataIterator
from generation import *
import random 


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

##############################################################################
# parameters
BATCH_SIZE = 512 # Batch size
MAX_STEPS = 300 # maximum length of your sequence
ITERS = 30000 # how many generator iterations to train for
REG = 0.1 # tradeoff between time and mark loss
LR = 1e-3 # learning rate
TYPE = sys.argv[1] # model type: joint event timeseries
NUM_steps_timeseries = 7 # timeseries steps before one event
Timeseries_feature = 4 #  time series feature size

SEED = 42345 # set graph-level seed to make the random sequences generated by all ops be repeatable across sessions
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)


############################e##################################################
# prepare data

#this is just toy data to test the code.
DIM_SIZE = 7 # equal num of classes
mi = MarkedIntensityHomogenuosPoisson(DIM_SIZE)
for u in range(DIM_SIZE):
    mi.initialize(1.0, u)
simulated_sequences = generate_samples_marked(mi, 15.0, 1000) # SHAPE = 1000 x (random) x 2
event_iterator = PaddedDataIterator(simulated_sequences,0,MARK=True,DIFF=True)
# time_series_data = np.ones((BATCH_SIZE,real_batch[0].shape[1],NUM_steps_timeseries,4))

###############################################################################
# define model

def RNNPP(
	num_classes=7, # number of dimensions for event sequence
	loss='mse', # loss type for time: mse and intensity, intensity loss comes from Du, etc. KDD16
	start = 3, # predict forward event starting at start-th event for each sequence
	reg = REG, # loss trade-off between mark and time
	state_size_event = 16, # RNN state size for event sequence
	state_size_timeseries = 32, #RNN state size for time series
	batch_size = BATCH_SIZE):


	epsilon = torch.tensor(1e-3).to(device)

	# linear layer
	if TYPE=='joint':
	    linear = nn.Linear(state_size_event+state_size_timeseries, 1, True).to(device)
	elif TYPE=='event':
	    linear = nn.Linear(state_size_event, 1, True).to(device)
	elif TYPE=='timeseries':
	    linear = nn.Linear(state_size_timeseries, 1, True).to(device)

	w_t = torch.tensor(1).float().to(device)

	# Softmax layer
	if TYPE=='joint':
	    lstm_e = nn.LSTM(input_size=8, hidden_size=state_size_event, num_layers=1, batch_first=True).to(device)
	    lstm_t = nn.LSTM(input_size=4, hidden_size=state_size_timeseries, num_layers=1, batch_first=True).to(device)
	    optim = torch.optim.Adam(lstm_e.parameters(), lr=LR)
	    optim.add_param_group({"params":lstm_t.parameters()})
	    W_l = torch.rand((state_size_event+state_size_timeseries, num_classes)).to(device)
	elif TYPE=='event':
	    lstm_e = nn.LSTM(input_size=8, hidden_size=state_size_event, num_layers=1, batch_first=True).to(device)
	    optim = torch.optim.Adam(lstm_e.parameters(), lr=LR)
	    W_l = torch.rand((state_size_event, num_classes)).to(device)
	elif TYPE=='timeseries':
	    lstm_t = nn.LSTM(input_size=4, hidden_size=state_size_timeseries, num_layers=1, batch_first=True).to(device)
	    optim = torch.optim.Adam(lstm_t.parameters(), lr=LR)
	    W_l = torch.rand((state_size_timeseries, num_classes)).to(device)

	b_l = torch.zeros(num_classes).to(device)
	W_l.requires_grad = True
	b_l.requires_grad = True

	optim.add_param_group({"params":linear.parameters()})
	optim.add_param_group({"params":W_l})
	optim.add_param_group({"params":b_l})

	for it in range(ITERS):

		optim.zero_grad()

		if TYPE == 'joint':
			real_batch = event_iterator.next_batch(
				BATCH_SIZE)  # shape = tuple where [0] = BATCH_SIZE x (sequence len) x 2, [1] = BATCH_SIZE x 1
			time_series_data = np.ones(
				(BATCH_SIZE, real_batch[0].shape[1], NUM_steps_timeseries, 4))  # BATCH_SIZE x (sequence len) x 7 x 4
			rnn_inputs_timeseries = torch.from_numpy(time_series_data).float().to(device)

		if TYPE == 'event':
			real_batch = event_iterator.next_batch(BATCH_SIZE)

		if TYPE == 'timeseries':
			real_batch = event_iterator.next_batch(BATCH_SIZE)
			time_series_data = np.ones((BATCH_SIZE, real_batch[0].shape[1], NUM_steps_timeseries, 4))
			rnn_inputs_timeseries = torch.from_numpy(time_series_data).float().to(device)

		rnn_inputs_event = torch.from_numpy(real_batch[0]).to(device)
		seqlen = torch.from_numpy(real_batch[1]).to(device)

		num_steps = rnn_inputs_event.shape[1]
		event_size = rnn_inputs_event.shape[2]
		y = torch.cat([rnn_inputs_event[:, 1:, :], rnn_inputs_event[:, :1, :]], 1)
		y = torch.reshape(y, [-1, event_size])

		if TYPE=='joint' or TYPE=='event':
			# rnn for event sequence
			rnn_input_onehot = utils.to_one_hot(rnn_inputs_event, num_classes).to(device)
			rnn_inputs_event = torch.cat([rnn_input_onehot, rnn_inputs_event[:,:,1:].float()], 2)

        #reshape rnn_outputs
		if TYPE=='joint':
			rnn_outputs_event, final_state = lstm_e(rnn_inputs_event)
			rnn_inputs_timeseries = torch.reshape(rnn_inputs_timeseries,
				                                [-1, NUM_steps_timeseries, Timeseries_feature])
			rnn_outputs_timeseries, final_state = lstm_t(rnn_inputs_timeseries)
			rnn_outputs_timeseries = torch.reshape(rnn_outputs_timeseries[:,-1,:],[batch_size,num_steps,state_size_timeseries])
			rnn_outputs = torch.cat([rnn_outputs_event,rnn_outputs_timeseries], 2)
		elif TYPE=='event':
			rnn_outputs_event, final_state = lstm_e(rnn_inputs_event)
			rnn_outputs = rnn_outputs_event
		elif TYPE=='timeseries':
			rnn_inputs_timeseries = torch.reshape(rnn_inputs_timeseries,
				                                [-1, NUM_steps_timeseries, Timeseries_feature])  
			rnn_outputs_timeseries, final_state = lstm_t(rnn_inputs_timeseries)
			rnn_outputs_timeseries = torch.reshape(rnn_outputs_timeseries[:,-1,:],[batch_size,num_steps,state_size_timeseries])
			rnn_outputs = rnn_outputs_timeseries

		rnn_outputs_shape = rnn_outputs.shape
		rnn_outputs = torch.reshape(rnn_outputs, [-1, rnn_outputs_shape[-1]])

		if loss=='intensity':
			if w_t < epsilon:
				wt = torch.sign(w_t) * epsilon
			else:
				wt = w_t
			part1 = linear(rnn_outputs).reshape(-1)
			part2 = wt*y[:,1]
			time_loglike = part1 + part2 + (torch.exp(part1)-torch.exp(part1+part2))/wt
			time_loss = - time_loglike
		elif loss=='mse':
			time_hat = linear(rnn_outputs)
			time_loss = torch.abs(torch.reshape(time_hat,[-1]) - y[:,1])

		mark_logits = torch.matmul(rnn_outputs, W_l) + b_l


		mark_true = utils.to_one_hot_uni(y[:, 0], num_classes).to(device)
		loss_fun = torch.nn.BCEWithLogitsLoss(reduction='none')
		mark_loss = loss_fun(mark_logits, mark_true)
		mark_loss = mark_loss.mean(1)

		total_loss = mark_loss + reg*time_loss
		lower_triangular_ones = torch.tensor(np.tril(np.ones([MAX_STEPS, MAX_STEPS]))).float().to(device)

		#length of y minus 2 to drop last prediction
		seqlen_mask = lower_triangular_ones[seqlen -2][0:start+batch_size, start: start+num_steps-start]
		zeros_pad = torch.zeros([batch_size,start]).to(device)
		seqlen_mask = torch.cat([zeros_pad,seqlen_mask], 1)

		mark_loss = torch.reshape(mark_loss,[batch_size,num_steps])
		mark_loss *= seqlen_mask
		# Average over actual sequence lengths.
		mark_loss = torch.sum(mark_loss, dim=1)
		mark_loss = torch.mean(mark_loss)


		total_loss = torch.reshape(total_loss,[batch_size,num_steps])
		total_loss *= seqlen_mask  #why 256*256 vs 256*140
		# Average over actual sequence lengths.
		total_loss = torch.sum(total_loss, dim=1)
		total_loss = torch.mean(total_loss)

		time_loss = total_loss - mark_loss
		total_loss.backward()

		optim.step()

		print(
			'Iter: {};  Total loss: {:.4};  Mark loss: {:.4};  Time loss: {:.4}'.format(it, total_loss, mark_loss,
                                                                                        time_loss))

	return total_loss,mark_loss,time_loss


RNNPP()

