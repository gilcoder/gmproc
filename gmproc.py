
import multiprocessing as mp
import time

class Workers:
	def __init__(self):
		self.targets = {}
		self.queue = mp.Queue()
		self.results = {}

	def add(self, id, target, params=None):
		self.targets[id] = ProcessWrapper(id, target, params)

	def set_params(self, id, new_value):
		self.targets[id].params = new_value

	def run(self, ids=None):
		if ids is None:
			ids = self.targets.keys()

		for k in ids:
			p = self.targets[k]
			pr = mp.Process(target=p.run, args=(p.id, self.queue, p.params))
			pr.start()
			id, value = self.queue.get()
			self.results[id] = value
		pr.join()
		return self.results


class ClientServer:
	def __init__(self, server, clients_delay = 0.1):
		self.targets = {}
		self.queue = mp.Queue()
		self.results = {}
		self.server = ServerWrapper(server)
		self.clients_delay = 0.1

	def add(self, id, client, params=None):
		self.targets[id] = ClientWrapper(id, client, params)

	def set_params(self, id, new_value):
		self.targets[id].params = new_value

	def run(self, ids=None):
		if ids is None:
			ids = self.targets.keys()
		cqueue = {}
		
		for k in ids:
			p = self.targets[k]
			pr = mp.Process(target=p.run, args=(p.id, p.queue, self.queue, p.params))
			cqueue[p.id] = p.queue
			p.process = pr

		ps = mp.Process(target=self.server.run, args=(cqueue, self.queue))
		ps.start()
		time.sleep(self.clients_delay)

		for k in ids:
			c = self.targets[k]
			if c.process is not None:
				c.process.start()

		ps.join()

		for k in ids:
			c = self.targets[k]

		for key in cqueue.keys():
			cqueue[key].close()
			if c.process is not None:
				c.process.join()
				c.queue.close()
		self.queue.close()

class ProcessWrapper:
	def __init__(self, id, target, params=None):
		self.id = id
		self.target = target
		self.params = params
	
	def run(self, id, queue, params):
		value = self.target(params)
		queue.put( (id, value) )


class ClientWorker:
	def __init__(self):
		pass

	def start(self):
		pass

	def process(self):
		pass

	def update(self):
		pass

	def wait(self):
		pass

	def finish(self):
		pass

	def done(self):
		return False

class ServerWorker:
	def __init__(self):
		pass

	def start(self):
		pass

	def process(self, id, msg):
		return None

	def wait(self):
		pass

	def finish(self):
		pass

	def done(self):
		return False


class ServerWrapper:
	def __init__(self, target):
		self.target = target

	def run(self, cqueue, squeue):
		self.target.start()
		while not self.target.done():
			id, msg = squeue.get()
			response = self.target.process(id, msg)
			if response is not None:
				cqueue[id].put(response)
			self.target.wait()
		self.target.finish()

class ClientWrapper(ProcessWrapper):
	def __init__(self, id, target, params=None):
		super().__init__(id, target, params)
		self.queue = mp.Queue()
		self.process = None

	def run(self, id, cqueue, squeue, params):
		obj = self.target()
		obj.start()
		while not obj.done():
			msg = obj.process()
			squeue.put( (id, msg) )
			response = cqueue.get()
			obj.update(response)
			obj.wait()
		obj.finish()
