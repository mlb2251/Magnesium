import syncfs
import torch
import os
from syncfs import SharedObject

"""
Magnesium aims to be an ML manager that lets multiple processes work together, sharing knowledge about the results of using different hyperparameters etc. It's cool because you don't need to be running a server for it to work, instead every client works together to make sure they don't interfere with each other and they get the most up to date information. That's an impressive way of saying it's pretty much just a synchronous filesystem.


Splays: Loading big torch.save()d files is annoying, so what if each of the attributes were saved separately?

Here's the idea:

Magnesium basically handles saving and loading for you. It's the project manager so it handles --join --resume --new etc. It doesn't hold state like stats and stuff, it's more like an interface for saving/loading/coordinating.
You give it a save_obj which implements save() and mg.save() will simply call save() on it and write the result to disk somewhere. Your save method should return a dict of things to save. mg.save() might also be called with a splay=True arg (or View.splay may be True) in which case mg.save will call save() to get the dict of saveable things, but rather than saving them all in one file it'll save each piece in separate files. Any objects with ._splay=True defined will be recursively splayed. Classes can also impl mgsave()->None to indicate they should not be saved, even in nonsplay mode basically you go thru the __dict__ and exclude those from the ones the setup.mgsave() returns


mg.save_obj = Setup()
# splays let us do some cool partial-saving stuff:
mg.save('stats') # save only the setup.stats, in a nice low-memory manner.
mg.save('stats.otherthing') # save only the setup.stats, in a nice low-memory manner.
mg.quicksave() # save everything other than the state dicts in a splay


Mg does the lock {name}, write {name}.{pid}, mv {name}.{pid} -> {name}, unlock {name}

SEE ANY PHONE NOTES IVE ADDED

Should have a built in timer section so results get saved always and you can look back at old timings
Should record the cmdline args argv[]

blacklisting certain classes

Basically you have a Setup which just holds you model, dataloaders, optimizer, scheduler, plotter, hypers. It's essentially just a runtime container for those things and no longer deals with saving/loading except that it is ofc the target of both of those often.



"""

PROJ_FILE = 'projectfile'
EXPS_DIR =  'exps'

class Leaf:
    def __init__(self,val):
        self.val=val

class View:
    """
    A `View` acts like a connection in a server-client system, so it doesn't have a ton of state other than keeping track of what project/experiment you're dealing with. Pretty much whenever you query it for stuff it'll give you newly updated information from the database.
    """
    def __init__(self,root):
        self.fs = syncfs.SyncedFS(root)
        self.proj = Project(self.fs)
        self.exp = None
        self.sugg = None
        self.setup = None

        self.name = f"{exp}_{os.getpid()}"
    def set_setup(self, setup):
        self.setup = setup
    ## EXP CREATION/LOADING
    def new_exp(self, name):
        self._assert_has_exp()
        self.exp = self.proj.new_exp(name)
    def set_exp(self, name):
        self.exp = self.proj.get_exp(name)
    ## EXP MANAGEMENT
    def del_exp(self):
        self._assert_has_exp()
        self.proj.del_exp(self.exp.name)
        self.exp = None
    def temp(self):
        self._assert_has_exp()
        self.exp.temp = True
    ## SUGGESTIONS
    def get_sugg(self):
        self._assert_has_exp()
        self.sugg = self.exp.get_sugg()
    def valid_sugg(self):
        self._assert_has_exp()
        self._assert_has_sugg()
        return self.exp.valid_sugg(self.sugg)
    def del_sugg(self):
        self._assert_has_exp()
        self._assert_has_sugg()
        self.exp.del_sugg(self.sugg)
        self.sugg = None
    def get_open_suggs(self):
        self._assert_has_exp()
        return self.exps.get_open_suggs()
    def flush_suggs(self):
        self._assert_has_exp()
        self.exp.flush_suggs()
        if self.sugg is not None:
            self.sugg = None
    def close_sugg(self,loss,stats):
        self._assert_has_exp()
        self._assert_has_sugg()
        self.exp.close_sugg(self.sugg)
        self.sugg = None
    ## Argument parsing
    def parse_args(self):
        mg_config = get_arguments()
        target = mg_config.target

        # TODO do all the actual configuration stuff
        if mg_config.new:
            util.yellow(f"[--new] Creating new experiment '{target}'")
            self.new_exp(target)
            if mg_config.temp:
                self.temp()
            suggestor = suggestor_by_name(mg_config.suggestor[0])(mg_config.suggestor[1:])
            hypersconfig = HypersConfig(mg_config.hypers)
            #suggestor
            #hypersconfig
            pass
        if mg_config.join:
            util.yellow(f"[--join] Joining experiment '{target}'")
            pass
        if mg_config.resume:
            util.yellow(f"[--resume] Resuming experiment '{target}'")
            pass
        if mg_config.control:
            util.yellow(f"[--control] Entering control mode")
            pass

        return mg_config.remainder
    ## Helper functions
    def _assert_has_exp(self):
        if self.exp is None:
            raise ValueError("View has not been assigned an exp so you can't call exp operations on it")
    def _assert_has_sugg(self):
        if self.sugg is None:
            raise ValueError("View has not been assigned a sugg so you can't call exp operations on it")

"""
VERY important. The rules for BoundObjects and mutating SharedObjects
If `self` is a SharedObject:
    Any degree of getattr is ok: self.x.y.z[3].n().z
    One degree of setattr is ok: self.x = y
    A setattr is ok for a BoundObject: self.bound_obj.x = y
    A setitem is ok for a BoundObject: self.bound_obj[x] = y
    A mutating fn call is ok for a BoundObject: self.bound_obj.pop(3)
    ## STUFF THATS NOT OK ##
    self.x.y.z = 3 # NEVER ok
    self.x.y = 3 # ONLY ok if x is bound
    self.x[y] = 3 # ONLY ok if x is bound
    self.bound_obj.y.z() # mutating functions must be direct functions of the boundobject, so if `y` were the called fn here it would work but since `z` is the called fn it doesn't work.

For anything more complicated than this just use the SharedObject.mod() contextmanager.
"""

# a magnesium directory is a project
class Project(SharedObject):
    def __init__(self,fs):
        path = os.path.join(fs.root,PROJ_FILE)
        super().__init__(path,fs)
    @self.lock()
    def new(self,name):
        super().new()
        self.name = name
        self.exps = {}
    @self.load
    def new_exp(self,name,hypers_config):
        if name in self.exps:
            raise ValueError(f"new_exp(): There is already an experiment named {name}")
        self.exps[name] = Experiment(name)
        return self.exps[name]
    @self.load
    def del_exp(self,name):
        if name not in self.exps:
            raise Exception(f"del_exp(): Experiment {name} not found")
        self.exps.pop(name)
    @self.load
    def get_exp(self,name):
        if name not in self.exps:
            raise Exception(f"get_exp(): Experiment {name} not found")
        return self.exps[name]
    @self.load
    def get_exps(self):
        return self.exps
    @self.load()
    def cleanup(self):
        for name,exp in self.exps.items():
            if exp.temp is True:
                self.exps.pop(name)


## boundobjects were a failure. Well they were successful but not dependable enough and not a nice enough interface
#class BoundObject:
#    """
#    A bound object is a simple object like a list or dict that is an attribute to a SharedObject. It is thus 'bound' to that shared object `self.parent` and the parent has it as an attriubte with the name `self.name`
#    Function calls like .pop() will be forwarded to the underlying data (e.g. a dict or list) and setupitem/delitem/setattr calls will
#    We do our best to ensure that update_parent gets called whenever getattr returns a callable function, however there could be other cases that mutate these objects so do not count on this! Always test the behavior! Remember you can always use the self.mod() contextmanager if bound objects dont work.
#    """
#    def __init__(self,name,parent,data):
#        assert isinstance(parent,SharedObject)
#        self.parent = parent
#        self.name = name
#        self.data = data
#
#    def update_parent(self):
#        setattr(self.parent,self.name,self)
#
#    # implement slicing and indexing
#    def __getitem__(self,*args,**kwargs):
#        return self.data.__getitem__(*args,**kwargs)
#    def __setitem__(self,*args,**kwargs):
#        self.data.__setitem__(*args,**kwargs)
#        self.update_parent()
#    def __delitem__(self,*args,**kwargs):
#        self.data.__delitem__(*args,**kwargs)
#        self.update_parent()
#
#    # this handles all the funciton calls like .pop()
#    def __getattr__(self,*args,**kwargs):
#        ret = getattr(self.dict,*args,**kwargs)
#        if callable(ret):
#            print("detected callable fn {ret}, wrapping in a ParentUpdate")
#            return ParentUpdate(ret,self)
#        print("non-callable value {ret}, we hopin it wont mutate bound_obj")
#        return ret
#    def __setattr__(self,*args,**kwargs):
#        setattr(self.dict,*args,**kwargs)
#        self.update_parent()
#
#
#class ParentUpdate:
#    """
#    A wrapper on a fn that calls `bound_obj.update_parent()` after running the function
#    """
#    def __init__(self,fn,bound_obj):
#        assert isinstance(bound_obj,BoundObject)
#        self.bound_obj = bound_obj
#        self.fn = fn
#    def __call__(self,*args,**kwargs):
#        ret = self.fn(*args,**kwargs)
#        self.bound_obj.update_parent()
#        print("Updated parent after calling {self.fn}")
#        return ret
#    # nobody really does get/setattr for functions but i guess ill be safe
#    def __getattr__(self,*args,**kwargs):
#        return getattr(self.fn,*args,**kwargs)
#    def __setattr__(self,*args,**kwargs):
#        setattr(self.fn,*args,**kwargs)
#
#class BoundList(BoundObject):
#    def __init__(self,parent,name):
#        super().__init__(parent,name,[])
#class BoundDict(BoundObject):
#    def __init__(self,parent,name):
#        super().__init__(parent,name,{})

# TODO a class decorator that add self.load() decorator to every function in a class, except for ones named in the class deco args (new and __init__)
class Experiment(SharedObject):
    def __init__(self,fs,name):
        path = os.path.join(fs.root,EXPS_DIR)
        super().__init__(path,fs)
    @self.lock()
    def new(self,name,hypers_config,suggestor):
        super().new()
        self.name = name # exp name, which is how the user will refer to this exp
        self.hypers_config = hypers_config # a HypersConfig object
        self.open_suggs = {} # open Suggestion objects
        self.closed_suggs = {} # closed Suggestion objects (ie loss has been reported)
        self.suggestor = suggestor # a Suggestor object
        self.temp = False # True means this can will be deleted by Project.cleanup()
    @self.load()
    def del_sugg(self,sugg_id):
        if sugg_id in self.open_suggs:
            self.open_suggs.pop(sugg_id)
        elif sugg_id in self.closed_suggs:
            self.closed_suggs.pop(sugg_id)
        else:
            raise Exception(f"del_sugg(): Suggestion {sugg_id} not found")
    @self.load()
    def get_open_suggs(self):
        return self.open_suggs
    @self.load()
    def get_closed_suggs(self):
        return self.closed_suggs
    @self.load()
    def valid_sugg(self,sugg_id):
        return (sugg_id in self.suggs)
    @self.load()
    def get_sugg(self):
        sugg = self.suggestor.get_sugg(self)
        return sugg
    @self.load()
    def flush_suggs(self):
        self.open_suggs = BoundDict('exps',self)
    @self.load()
    def close_sugg(self,sugg_id,loss,stats):
        if sugg_id not in self.open_suggs:
            raise Exception(f"close_sugg(): Suggestion {sugg_id} not in open_suggs")
        self.open_suggs[sugg_id].close(loss,stats)
        self.close_suggs[sugg_id] = self.open_suggs.pop(sugg_id)

class Suggestion:
    def __init__(self, id, hypers, exp):
        self.id = id
        self.hypers = hypers
        self.loss = None
        self.stats = None
    def close(self,loss,stats=None):
        self.loss = loss
        self.stats = stats


class Suggestor:
    def __init__(self,bandwidth):
        self.bandwidth = bandwidth
    def get_sugg(self,exp):
        raise NotImplementedError

class SigoptSuggestor(Suggestor):
    def __init__(self,bandwidth):
        super().__init__(bandwidth)
    def get_sugg(self,exp):
        pass

class SigoptExperiment(Experiment):
    pass

class SigoptProject(Project):
    pass



def bind(dir,project=None,exp=None):
    """
    Create a new Magnesium session, same as Magnesium() but `bind` is a cool name so we use it.
    """
    return Magnesium(dir,project,exp)

def get_arguments():
    parser = argparse.ArgumentParser(description='MPNN for J-Coupling Kaggle competition')
    ## Exclusive arguments (must use exactly one of them)
    parser.add_argument('--new', metavar='name',
                        help='Start a new experiment with the specified name')
    parser.add_argument('--join', metavar='exp_file [new_file]', nargs='+',
                        help='Join an existing experiment with the specified name')
    parser.add_argument('--resume', metavar='name [device]',  nargs='+',
                        help='Resume an experiment. Restarts at the last epoch or at the next step of hyper optimization. Optionally provide a device number or \'cpu\' as a second argument and data will be transferred to this device instead of wherever it used to be stored')
    parser.add_argument('--control', action='store_true', # we set default to None not False so that `exclusive_options` works below
                        help='Enter the control console for the project')
    ## Non-exclusive arguments
    parser.add_argument('--suggestor', nargs='+',
                        help='Only used with --new. Enter the suggestor type followed by any arguments for the suggestor (no internal "--", use "=" between key and value, and no spaces within a key/value pair). Example: --suggestor sigopt token=REAL bandwidth=4 budget=1000')
    parser.add_argument('--hypers', nargs='+',
                        help='Only used with --new. Enter any hypers you want to overrides the defaults for (no internal "--", use "=" between key and value, and no spaces within a key/value pair). Example: --hypers bins=20 message_len=160')
    parser.add_argument('--temp', action='store_true',default=False,
                        help='For use with `new` to create a temp project')
    parser.add_argument('remainder', nargs=argparse.REMAINDER)

    mg_config = parser.parse_args()

    # Validity assertions
    exclusive_options = ['new','join','resume','control']
    selected_opt = getattr(mg_config,x) is not None for x in exclusive_options] # there should be one True in here at the index in exclusive_options for the option that was selected
    assert sum(selected_opt) == 1, f"You must provide exactly one of the options: {exclusive_options}"
    mg_config.target = exclusive_option[selected_opt.index(True)] # the args for whatever the selected option was
    if mg_config.suggestor is not None:
        assert mg_config.new is not None, "If you provide --suggestor you must be starting a --new experiment"
    if mg_config.temp is True:
        assert mg_config.new is not None, "If you provide --temp you must be starting a --new experiment"
    if mg_config.hypers is not None:
        assert mg_config.new is not None, "If you provide --hypers you must be starting a --new experiment"

    return mg_config


def suggestor_by_name(name):
    if name == 'sigopt':
        return SigoptSuggestor
    raise NotImplementedError


class Setup:
    def __init__(self,config):




        if config.join is not None:
            setup_dict = torch.load(fpath(config.join[0],'r'))
            self._load(setup_dict)
            self.file = fpath(config.join[1],'w',config)
            self.expname = expname(config.join[0])
            util.purple(f"[--join] Joining existing experiment '{config.join[0]}' saving results to '{expname(self.file)}'")
            self.mode = 'join' # must do after _load
            if config.device == setup_dict['raw']['config'].device:
                util.red(f"Warning: device {config.device} is also being used by the experiment you are `--join`ing off of")
            self.config = Setup.config_override(setup_dict,config)
            self.stats = Stats()
            self.trainloop = False
            self.has_suggestion = False
            return
        elif config.resume is not None:
            util.green(f"[--resume] Resuming model '{config.resume[0]}'")
            device = None
            if len(config.resume) > 1:
                device = torch.device(int(config.resume[1]) if (config.resume[1] != 'cpu') else 'cpu')
                util.green(f"[--resume] Transferring model to {device}")
            setup_dict = torch.load(fpath(config.resume[0],'r'),map_location=device)
            self._load(setup_dict) # load attrs from old setup
            self.mode = 'resume' # must do after _load
            self.config = Setup.config_override(setup_dict,config)

            if device is not None: # override with the [--resume filename [device]]
                self.config.device = device

            # only for 'resume' mode
            if self.trainloop is True:
                util.yellow("Previously halted in trainloop, loading state dicts")
                self._old_state_dicts = setup_dict['state_dicts']

            self.stats = Stats(self.stats) # update stats to the newest version

            if self.stats.curr_epoch >= self.config.epochs:
                self.config.epochs = self.stats.curr_epoch + self.config.epochs
                print(f"Auto-adjusting max epochs to {self.config.epochs}")
            print(f"Model will run from epoch {self.stats.curr_epoch}->{self.config.epochs}")
            return
        elif config.del_exp is not None:
            setup_dict = torch.load(fpath(config.del_exp,'r'))
            self._load(setup_dict)
            conn = sigopt_utils.Conn(self)
            conn.del_exp()
            util.green("deleted exp")
            self.save()
            exit(0)
        elif config.del_sugg is not None:
            setup_dict = torch.load(fpath(config.del_sugg,'r'))
            self._load(setup_dict) # load attrs from old setup
            conn = sigopt_utils.Conn(self)
            conn.del_sugg()
            util.green("deleted suggestion")
            self.save()
            exit(0)
        elif config.inspect is not None:
            if config.inspect.endswith('.stats'):
                stats = torch.load(fpath(config.inspect,'r'))
                util.green('Loaded stats into `stats`')
                breakpoint()
                exit(0)
            try:
                setup_dict = torch.load(fpath(config.inspect,'r'))
            except:
                setup_dict = torch.load(fpath(config.inspect,'r'),map_location=torch.device('cpu'))
            self._load(setup_dict) # load attrs from old setup
            setup = self
            util.green('Loaded setup into `self` and `setup`')
            breakpoint()
            exit(0)
        else:
            if config.new is not None:
                util.yellow(f"[--new] Creating new model named '{config.new}'")
                self.file = fpath(config.new,'w',config)
            else:
                util.yellow(f"Creating new unsaved model")
                self.file = None
            self.expname = expname(self.file)
            self.mode = 'new'
            self.config = config
            self.trainloop = False
            self.has_suggestion = False
            self.stats = Stats()
            return

    def _load(self,setup_dict):
        for key,val in setup_dict['raw'].items():
            self[key] = val

    def load_state_dicts(self):
        """
        Load model/optimizer/scheduler state dicts if we're resuming an old session
        """
        for key,val in self._old_state_dicts.items():
            self[key].load_state_dict(val)
        del self._old_state_dicts # so we don't save it again and take up a ton of space

    def __getitem__(self,key):
        return getattr(self,key)
    def __setitem__(self,key,val):
        return setattr(self,key,val)
    def __repr__(self):
        body = []
        for k,v in self.__dict__.items():
            body.append(f"{k}={v}")
        body = '\n'+'\n'.join(body)+'\n'
        return f"Setup({body})"

    def save(self):
        """
        Save everything!
        """
        blacklist = ['train_loader','valid_loader','test_loader','conn', 'plt','policy']
        if self.file is None:
            return
        setup_dict = {'raw':{}, 'state_dicts':{}}
        for k,v in self.__dict__.items():
            if k in blacklist:
                continue
            if hasattr(v,'state_dict'):
                setup_dict['state_dicts'][k] = v.state_dict()
            else:
                setup_dict['raw'][k] = v
        util.gray(f"saving raw: {list(setup_dict['raw'].keys())}")
        torch.save(setup_dict,f"{self.file}")
        torch.save(self.stats,f"{self.file}.stats")
        util.gray(f"[saved to {self.file}]")

    @staticmethod
    def config_override(oldsetup_dict, newconfig):
        """
        Overrides certain fields in `oldsetup.config` using ones from `newconfig` as specified in `newconfig.override` and `newconfig.override_all`
        Note that in reality we actually copy all fields from `old` into `new` as long as override is not specified. This allows better compatibility with fields added to the program that are present in newconfig but not oldconfig.
        """
        oldconfig = oldsetup_dict['raw']['config']
        if newconfig.device != torch.device(0): # TODO this should be generally done for all nondefault things. Maybe argparse has a way to telling which args were specified and which werent. otherwise just do it for the things that no longer match whatever their default was
            newconfig.override.append('device')
        newconfig.override.append('override')
        newconfig.override.append('override_all')
        if newconfig.override_all:
            return newconfig
        for key in oldconfig.__dict__:
            if key not in newconfig.override:
                setattr(newconfig,key,getattr(oldconfig,key))
            else:
                util.yellow(f"overriding {key}")
        return newconfig




"""
A class representing a single hyperparameter. Used by Hypers.
"""
class Param:
    def __init__(self,name,*,default='_not_set',max=None,min=None,type='int',tunable=True,derive_fn=None, opts=None):
        assert name is not None, "You must provide a name for the parameter"
        if (type == 'int') and (isinstance(default,float) or isinstance(max,float) or isinstance(min,float)):
            raise Exception(f"Warning: casting float default/max/min of Param '{name}' to int. You will receive all suggestions as ints for '{name}'. Use Param(type='float') to avoid this behavior")
        self.name=name
        self.min=min
        self.max=max
        self.type=type
        self.tunable = tunable if (derive_fn is None) else False # always False if derive_fn provided
        self.val=default
        self.derive_fn=derive_fn
        self.opts = opts # options for categorical values

        if derive_fn: # we just always set our value to the result of derive_fn(hypers)
            if isinstance(derive_fn,str):
                other_name = derive_fn
                derive_fn = CommonCase(other_name,self.name)
            self.derive_fn = derive_fn
            self.derived = True
            return

        self.derived = False
        assert default != '_not_set', f"You must provide a default value for the Param `{name}` since it is not a derived value"
        if tunable:
            if self.type not in ['categorical','bool']:
                assert max is not None and min is not None, f"You must provide both max and min values when creating the non-categorical tunable Param `{name}`"
            elif self.type == 'categorical':
                assert self.opts is not None, f"You must provide `opts` when creating the categorical tunable Param `{name}`"
    def __repr__(self):
        return f"{self.name}={self.val}"

class CommonCase: # since multiline closures aren't pickleable but classes are
    def __init__(self,other_name,name):
        self.other_name = other_name
        self.name = name
    def __call__(self,hypers):
        assert self.other_name in hypers.keys, f"Unable to derive `{self.name}` from value `{self.other_name}` because `{self.other_name}` not found in the hypers list"
        assert hypers.keys.index(self.other_name) < hypers.keys.index(self.name),"{self.name} is derived from {self.other_name} but is defined before {self.other_name}! Please move the derived value after the source value"
        return hypers[self.other_name].val

class Hypers:
    def __init__(self,from_dict=None):
        if from_dict is not None:
            for k,v in from_dict.items():
                self[k] = v
    def __getitem__(self,key):
        return getattr(self,key)
    def __setitem__(self,key,val):
        setattr(self,key,val)
    def get(self,key,layerno=None):
        if layerno is not None:
            return self[key+'_'+str(layerno)]
        return self[key]
    def __repr__(self):
        body = []
        for k,v in self.__dict__.items():
            body.append(f"{k}={v}")
        body = ','.join(body)
        return f"Hypers({body})"

"""
Be sure to call hypers.finalize() before using any hypers if you manually change things around. Finalize() is automatically called for you: 1) at the end of __init__, 2) at the start of get_sigopt_parameters, and 3) at the end of assign_parameters.
IMPORTANT: derived values must come AFTER the values that they are derived from in the hypers list
"""
class HypersConfig:
    def __init__(self,param_list):
        self.keys = []
        self.tunable_params = []
        self.params = param_list
        for param in param_list:
            assert not hasattr(self,param.name)
            self[param.name] = param
            self.keys.append(param.name)
            if param.tunable:
                self.tunable_params.append(param)
        self.finalize()

    def get_hypers(self):
        self.finalize()
        return Hypers({k:self[k].val for k in self.keys})

    def finalize(self):
        for k in self.keys:
            if self[k].derived:
                self[k].val = self[k].derive_fn(self)

    def get_sigopt_parameters(self):
        self.finalize()
        sig_params = []
        for param in self.tunable_params:
            ty = param.type
            if ty == 'float':
                ty = 'double'
            if ty == 'categorical':
                opts = param.opts
            if ty == 'bool':
                ty = 'categorical'
                opts = ['True','False']

            if ty == 'categorical':
                sig_params.append(dict(name=param.name,type=ty,categorical_values=opts))
            else:
                sig_params.append(dict(name=param.name,type=ty,bounds=dict(min=param.min,max=param.max)))
        return sig_params

    def assign_parameters(self,assignments):
        for name,val in assignments.items():
            if self[name].type == 'bool':
                val = {'True':True,'False':False}[val]
            self[name].val = val
        self.finalize()

    def __getitem__(self,key):
        return getattr(self,key)
    def __setitem__(self,key,val):
        return setattr(self,key,val)
    def __repr__(self):
        body = []
        for k in self.keys:
            body.append(repr(self[k]))
        body = ','.join(body)
        return f"HypersConfig({body})"

"""
Stats for a single train_loop() call (ie one stats object per configuration of hypers)
"""
class Stats:
    def __init__(self,old=None, policy=None):
        self.train_losses = []
        self.valid_losses = []
        self.accuracy = []
        #self.examples = [] # SMILES string outputs
        self.timestamps = [] # useful for figuring out if a model was paused then resumed
        self.config = None
        self.hypers = None
        self.curr_epoch = 0
        self._magic = 'stats' # useful since `Stats` from an old version might not count as `Stats` from a new version in terms of isinstance()
        self.policy = policy

        # load values from `old`. Throw out anything that's from an old version tho (ie only look for keys in self.__dict__)
        if old is not None:
            if old._magic == 'save':
                old = old.stats
            if old._magic == 'stats': # if NOT elif
                for key in self.__dict__:
                    if hasattr(old,key):
                        self[key] = old[key]
    def __getitem__(self,key):
        return getattr(self,key)
    def __setitem__(self,key,val):
        return setattr(self,key,val)
    def __repr__(self):
        body = []
        for k,v in self.__dict__.items():
            body.append(f"{k}={v}")
        body = '\n'+'\n'.join(body)+'\n'
        return f"Stats({body})"

    def verify(self):
        try:
            if len(self.train_scores) != len(self.timestamps):
                print("[warn] stats._verify() warning: len(self.train_scores) != len(self.timestamps)")
            if len(self.valid_scores) != len(self.timestamps):
                print("[warn] stats._verify() warning: len(self.valid_scores) != len(self.timestamps)")
            if len(self.accuracy) != len(self.timestamps):
                print("[warn] stats._verify() warning: len(self.accuracy) != len(self.timestamps)")
            if len(self.examples) != len(self.timestamps):
                print("[warn] stats._verify() warning: len(self.examples) != len(self.timestamps)")
            if self.curr_epoch != len(self.timestamps):
                print("[warn] stats._verify() warning: self.curr_epoch != len(self.timestamps)")
        except AttributeError as e:
            print(f"error during verify() because of {e}, unable to complete verify(). ignoring and continuing")
    def check_abort(self):
        if self.policy is None:
            return False
        return self.policy.check_abort(self)

"""
A policy embedded in `stats` for deciding when to end early. check_abort takes a Stats object and returns True if it should abort. This gets called at the end of trainloop right after stats.curr_epoch+=1 happens so curr_epoch == whatever the next epoch to be potentially run is.
"""
class Policy:
    def check_convergence(self):
        raise NotImplementedError

"""
A simple policy of ending after `num_epochs` have been run
"""
class EpochPolicy(Policy):
    def __init__(self,num_epochs,stats):
        self.stats = stats
        self.num_epochs = num_epochs
    def check_convergence(self):
        if self.stats.curr_epoch < self.num_epochs:
            return False
        return True


class PlateauPolicy(Policy):
    def __init__(self,sched):
        """
        Takes a ReduceLROnPlateau scheduler and converges when the scheduler brings the learning rate down to its min value.
        """
        assert type(sched) is torch.optim.lr_scheduler.ReduceLROnPlateau
        self.sched = sched
    def check_convergence(self):
        """
        This code is modeled from torch.optim.lr_scheduler.ReduceLROnPlateau._reduce_lr
        """
        for i,param_group in enumerate(self.sched.optimizer.param_groups):
            if param_group['lr'] > self.sched.min_lrs[i]:
                print("curr lr:",param_group['lr'])
                return False
        return True









