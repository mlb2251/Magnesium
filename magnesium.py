import syncfs
import torch
import os
from syncfs import SharedObject

"""
Magnesium aims to be an ML manager that lets multiple processes work together, sharing knowledge about the results of using different hyperparameters etc. It's cool because you don't need to be running a server for it to work, instead every client works together to make sure they don't interfere with each other and they get the most up to date information. That's an impressive way of saying it's pretty much just a synchronous filesystem.


Splays: Loading big torch.save()d files is annoying, so what if each of the attributes were saved separately?

Here's the idea:

Magnesium basically handles saving and loading for you. It's the project manager so it handles --join --resume --new etc. It doesn't hold state like stats and stuff, it's more like an interface for saving/loading/coordinating.
You give it a save_obj which implements save() and mg.save() will simply call save() on it and write the result to disk somewhere. Your save method should return a dict of things to save. mg.save() might also be called with a splay=True arg (or View.splay may be True) in which case mg.save will call save() to get the dict of saveable things, but rather than saving them all in one file it'll save each piece in separate files. Any objects with ._splay=True defined will be recursively splayed. Classes can also impl mgsave()->None to indicate they should not be saved, even in nonsplay mode basically you go thru the __dict__ and exclude those from the ones the session.mgsave() returns


mg.save_obj = session()
# splays let us do some cool partial-saving stuff:
mg.save('stats') # save only the session.stats, in a nice low-memory manner.
mg.save('stats.otherthing') # save only the session.stats, in a nice low-memory manner.
mg.quicksave() # save everything other than the state dicts in a splay

SEE ANY PHONE NOTES IVE ADDED

Should have a built in timer section so results get saved always and you can look back at old timings
Should record the cmdline args argv[]

blacklisting certain classes

Basically you have a session which just holds you model, dataloaders, optimizer, scheduler, plotter, hypers. It's essentially just a runtime container for those things and no longer deals with saving/loading except that it is ofc the target of both of those often.





"""

PROJ_FILE = 'projectfile'
EXPS_DIR =  'experiments'
SESS_DIR =  'sessions'

class Leaf:
    def __init__(self,val):
        self.val=val

class View:
    """
    A `View` acts like a connection in a server-client system, so it doesn't have a ton of state other than keeping track of what project/experiment you're dealing with. Pretty much whenever you query it for stuff it'll give you newly updated information from the database.

    ** We don't save View to disk anywhere. Sess holds any data that's useful to View and needs to be saved. The point of View is that it's very simple and lightweight but holds many of the API functions for interacting with experiments, projects, and suggestions. **

    .fs:   the current SyncedFS
    .proj: the current Project. This is a SharedObject so it is automatically synchronized across processes and never needs to be saved explicitly.
    .exp:  the current Experiment. Also a SharedObject, so automatically synchronized and saved.
    .sugg: the current Suggestion. This is actually an alias (via @property) of .sess.sugg, and setting/getting this one will actually set/get .sess.sugg. We don't need to save this every because there's a copy in .sess that will be saved when we save .sess
    .sess: the current Session. This is the one thing that needs to be saved
    """
    def __init__(self,root):
        self.fs = syncfs.SyncedFS(root)
        self.proj = Project(self.fs,'any',name) # current Project
        self.exp = None # current Experiment
        self.sugg = None # current Suggestion
        self.sess = None # current Session
    ## SESS CREATION/LOADING
    def new_sess(self,name,config):
        """
        Create a new Session from the given config dict, which must include all possible config keys to be valid.
        """
        config = Config(config)
        self.sess = self.exp.new_sess(name,config)
    def load_sess(self,name,config_override):
        """
        Load an existing Session from disk, then update its Config with values in `config_override`. `config_override` does not need to have all possible config keys, it should only have those you want to override.
        If 'device' is a key in config_override the Session will be loaded onto the specified device rather than whatever device it was originally on (uses the torch.load `map_location` kwarg under the hood).
        Sets the loaded session to the current session, and sets the current suggestion to the one present in the loaded Session.
        """
        self.sess = self.exp.load_sess(name,config_override)
        self.sugg = self.sess.sugg

    ## EXP CREATION/LOADING
    def new_exp(self,name,suggestor,hypersconfig):
        """
        Create a new experiment named `name` with Suggestor `suggestor` and HypersConfig `hypersconfig`.
        Will throw an error if an experiment by this name already exists.
        Sets current experiment to the new experiment.
        """
        self._assert_has_exp()
        self.exp = self.proj.new_exp(name,suggestor,hypersconfig)
    def set_exp(self, name):
        """
        Set the current experiment to be the experiment `name` (this experiment must already exist)
        """
        self.exp = self.proj.get_exp(name)
    ## EXP MANAGEMENT
    def del_exp(self):
        """
        Delete the current experiment
        """
        self._assert_has_exp()
        self.proj.del_exp(self.exp.name)
        self.exp = None
    def temp(self):
        """
        Mark the current experiment as temporary, so that if View.cleanup() is ever called it will be deleted.
        """
        self._assert_has_exp()
        self.exp.temp = True
    ## SUGGESTIONS
    def get_sugg(self):
        """
        Get a new suggestion from the current Experiment's Suggestor. This will be added to the experiment's open suggestions.
        """
        self._assert_has_exp()
        self.sugg = self.exp.get_sugg()
    def valid_sugg(self):
        """
        Boolean indicating if the current Suggestion is still an open suggestion of the current Experiment
        """
        self._assert_has_exp()
        self._assert_has_sugg()
        return self.exp.valid_sugg(self.sugg)
    def del_sugg(self):
        """
        Delete the current suggestion. Works for both closed and open suggestions. Error if suggestion isn't in open or closed suggestions.
        """
        self._assert_has_exp()
        self._assert_has_sugg()
        self.exp.del_sugg(self.sugg)
        self.sugg = None
    def get_open_suggs(self):
        """
        Returns a list of Suggestion objects which are the currently open suggestions for the current experiment.
        """
        self._assert_has_exp()
        return self.exps.get_open_suggs()
    def flush_suggs(self):
        """
        Remove all open suggestions from the experiment
        Also removes current suggestion, if any.
        """
        self._assert_has_exp()
        self.exp.flush_suggs()
        if self.sugg is not None:
            self.sugg = None
    def close_sugg(self,loss,stats):
        """
        Report on the results of trying out a suggestion to close it. It will be moved from open suggestions to closed suggestions for the current experiment.
        Removes current suggestion.
        """
        self._assert_has_exp()
        self._assert_has_sugg()
        self.exp.close_sugg(self.sugg)
        self.sugg = None
    ## Argument parsing
    def parse_args(self):
        """
        This is a very important function for all of Magnesium. It handles whatever initial loading, setting, and object creation needs to be done based on the user's command line arguments. View gets configured with an (initial) project/experiment/suggestion/session along the way, if applicable.
        """
        mg_config = get_arguments()
        target = mg_config.target # this gets set to whatever the value of the 'exclusive' option used is, for example for --new it would be [expname,sessname,device]
        mode = mg_config.mode

        sess_config_kwargs = kwargs_of_strargs(mg_config.session,conversion_dict=sess_config_types)
        for kwarg in sess_config_kwargs:
            assert kwargs != 'device', "device can only be specified as a positional argument in --new --join or --resume"
            assert kwarg in get_default_sess_config().keys(), f"{kwarg} not in get_default_sess_config(), you should update it if this is a valid kwarg" 
            assert kwarg in allowed_sess_config[mode], f"--session argument {kwarg} is not allowed for --{mode}"

        if mode == 'new': ## --new
            expname,sessname,device = target
            util.yellow(f"[--new] Creating new experiment '{expname}' with session '{sessname}' on device '{device}'")
            if mg_config.temp:
                self.temp()
            suggestor = suggestor_by_name(mg_config.suggestor[0])(mg_config.suggestor[1:])
            hypersconfig = HypersConfig(mg_config.hypers)
            self.new_exp(expname,suggestor,hypersconfig) # new exp

            # building sess_config
            sess_config = get_default_sess_config()
            sess_config.update(sess_config_kwargs) # override defaults
            sess_config['device'] = device_of_str(device)
            self.new_sess(sessname,sess_config) # new sess

        elif mode == 'join': ## --join
            expname,sessname,device = target
            util.yellow(f"[--join] Joining experiment '{expname}' with session '{sessname}' on device '{device}'")
            self.set_exp(expname) # old exp

            # building sess_config
            assert exp.defaults_sess_config is not None, "Can't join an experiment if no other Sessions have been started in it!"
            sess_config = {k:v for k,v in exp.defaults_sess_config.items()} # deep enough copy
            sess_config.update(sess_config_kwargs) # override defaults
            sess_config['device'] = device_of_str(device)
            self.new_sess(sessname,sess_config) # new sess

        elif mode == 'resume': ## --resume
            if len(target) == 3: # a device was specified
                expname,sessname,device = target
                sess_config_kwargs['device'] = device_of_str(device)
                util.yellow(f"[--resume] Resuming session '{sessname}' of experiment '{expname}' on device '{device}'")
            else: # no device specified
                expname,sessname = target
                util.yellow(f"[--resume] Resuming session '{sessname}' of experiment '{expname}'")
            self.set_exp(expname) # old exp

            # loading sess
            self.load_sess(sessname,config_override=sess_config_kwargs) # old sess
        elif mode == 'control': ## --control
            util.yellow(f"[--control] Entering control mode")
            pass
        else:
            raise Exception("Unrecognized mode")

    ## Helper functions
    def _assert_has_exp(self):
        if self.exp is None:
            raise ValueError("View has not been assigned an exp so you can't call exp operations on it")
    def _assert_has_sugg(self):
        if self.sugg is None:
            raise ValueError("View has not been assigned a sugg so you can't call exp operations on it")

#"""
#VERY important. The rules for BoundObjects and mutating SharedObjects
#If `self` is a SharedObject:
#    Any degree of getattr is ok: self.x.y.z[3].n().z
#    One degree of setattr is ok: self.x = y
#    A setattr is ok for a BoundObject: self.bound_obj.x = y
#    A setitem is ok for a BoundObject: self.bound_obj[x] = y
#    A mutating fn call is ok for a BoundObject: self.bound_obj.pop(3)
#    ## STUFF THATS NOT OK ##
#    self.x.y.z = 3 # NEVER ok
#    self.x.y = 3 # ONLY ok if x is bound
#    self.x[y] = 3 # ONLY ok if x is bound
#    self.bound_obj.y.z() # mutating functions must be direct functions of the boundobject, so if `y` were the called fn here it would work but since `z` is the called fn it doesn't work.
#
#For anything more complicated than this just use the SharedObject.mod() contextmanager.
#"""

# a magnesium directory is a project
class Project(SharedObject):
    def __init__(self,fs,mode,name):
        path = os.path.join(fs.root,PROJ_FILE)
        super().__init__(path,fs,mode,    name)
    @self.lock()
    def new(self,name):
        """
        The new() function gets called (by SharedObject.__init__ usually) when the shared object gets completely wiped and reset, and also the very first time it is created.
        """
        super().new()
        self.name = name # unique Experiment name
        self.exps = {} # expname->Experiment dict with all experiments
    @self.load
    def new_exp(self,name,suggestor,hypers_config):
        """
        Called by View.new_exp(). Creates a new Experiment and adds it to `self.exps`
        """
        if name in self.exps:
            raise ValueError(f"new_exp(): There is already an experiment named {name}")
        self.exps[name] = Experiment(self.fs,'new',name,suggestor,hypers_config)
        return self.exps[name]
    @self.load
    def del_exp(self,name):
        """
        Called by View.del_exp(). Deletes an experiment by just poppin git from `self.exps`
        """
        if name not in self.exps:
            raise Exception(f"del_exp(): Experiment {name} not found")
        self.exps.pop(name)
    @self.load
    def get_exp(self,name):
        """
        Called by View.set_exp(). Retrieves an Experiment from `self.exps`
        """
        if name not in self.exps:
            raise Exception(f"get_exp(): Experiment {name} not found")
        return self.exps[name]
    @self.load
    def get_exps(self):
        """
        Return the dict of all expname->Experiment for all experiments in the project
        """
        return self.exps
    @self.load()
    def cleanup(self):
        """
        See View.temp() for reference. Project.cleanup() clears all Experiments marked as temporary.
        """
        for name,exp in self.exps.items():
            if exp.temp is True:
                self.exps.pop(name)


## boundobjects were a failure. Well they were successful but not dependable enough and not a nice enough interface
#class BoundObject:
#    """
#    A bound object is a simple object like a list or dict that is an attribute to a SharedObject. It is thus 'bound' to that shared object `self.parent` and the parent has it as an attriubte with the name `self.name`
#    Function calls like .pop() will be forwarded to the underlying data (e.g. a dict or list) and sessionitem/delitem/setattr calls will
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
    def __init__(self,fs,mode,name,suggestor,hypers_config):
        path = os.path.join(fs.root,EXPS_DIR)
        super().__init__(path,fs,mode,   name,suggestor,hypers_config) # gap indicates *args/**kwargs
    @self.lock()
    def new(self,name,suggestor,hypers_config):
        """
        New gets called when the Experiment instance is first instantiated, or what it gets wiped completely.
        """
        super().new()
        self.name = name # exp name, which is how the user will refer to this exp
        self.suggestor = suggestor # a Suggestor object
        self.hypers_config = hypers_config # a HypersConfig object
        self.open_suggs = {} # open Suggestion objects
        self.closed_suggs = {} # closed Suggestion objects (ie loss has been reported)
        self.open_sessions = [] # open sess_ids (includes ones that aren't actively running). We don't store the actual Session object bc thats huge and stored/updated separately. We only store the ids here.
        self.active_sessions = [] # sess_ids that are actively being trained on
        self.closed_sessions = [] # close sess_ids (ie they finished their Suggestion)
        self.temp = False # True means this can will be deleted by Project.cleanup()
        self.default_sess_config = None # the first Session created will set this, and all future sessions will use it as a default config.
    @self.load()
    def del_sugg(self,sugg_id):
        """
        Called by View.del_sugg(). Deletes a suggestion (open or closed)
        """
        if sugg_id in self.open_suggs:
            self.open_suggs.pop(sugg_id)
        elif sugg_id in self.closed_suggs:
            self.closed_suggs.pop(sugg_id)
        else:
            raise Exception(f"del_sugg(): Suggestion {sugg_id} not found")
    @self.load()
    def get_open_suggs(self):
        """
        Return all open suggestions in a dict from sugg_id->Suggestion
        """
        return self.open_suggs
    @self.load()
    def get_closed_suggs(self):
        """
        Return all closed suggestions in a dict from sugg_id->Suggestion
        """
        return self.closed_suggs
    @self.load()
    def valid_sugg(self,sugg_id):
        """
        Called by View.valid_sugg(). Checks if `sugg_id` is an open suggestion.
        """
        return (sugg_id in self.open_suggs)
    @self.load()
    def get_sugg(self):
        """
        Called by View.get_sugg(). Gets a new suggestion from the Suggestor.
        """
        sugg = self.suggestor.get_sugg(self)
        return sugg
    @self.load()
    def flush_suggs(self):
        """
        Called by View.flush_suggs(). Clear all open suggestions.
        """
        self.open_suggs = {}
    @self.load()
    def close_sugg(self,sugg_id,loss,stats):
        """
        Called by View.close_sugg(). Closes an open suggestion. The loss and stats get recorded by Suggestion.close() and the suggestion object gets moved to self.closed_suggs.
        """
        if sugg_id not in self.open_suggs:
            raise Exception(f"close_sugg(): Suggestion {sugg_id} not in open_suggs")
        self.open_suggs[sugg_id].close(loss,stats)
        self.close_suggs[sugg_id] = self.open_suggs.pop(sugg_id)

    @self.load()
    def new_sess(self,name,sess_config):
        """
        Create a new Session object. Sets `self.default_sess_config` if this is the first Session for the experiment. Adds the session id to `self.open_sessions`, and returns it.
        `sess_config`: a Config object
        Impl node: we only store session ids in self.open_sessions because the actual Session objects are huge since they contain the ML models, and if that were a local variable to us we would be writing it to disk constantly since we're a SharedObject.
        """
        if Session.get_id(self,name) in (self.open_sessions + self.closed_sessions):
            raise ValueError("This experiment alredy has a session by that name")
        if self.default_sess_config is None:
            assert len(self.open_sessions + self.closed_sessions) == 0
            # this is our first session so we use its args as default for all future sessions
            self.default_sess_config = sess_config.clone()
        sess = Session(name,sess_config,self.name)
        self.open_sessions.append(sess.id)
        return sess

    @self.load()
    def load_sess(self,name,config_override):
        """
        Called by View.load_sess(). Load a session object off of disk by name, and for any keys in the dict (NOT a Config) `config_override`, those key/value pairs will be overwritten in the loaded session's `.config`. If 'device' is in `config_override` then it will be used as the `map_location` keyword in torch.load so that the model in Session can be transferred to a new GPU if requested (can prevent crashes on loading a gpu-saved thing on a computer without a gpu for exmaple, by allowed you to load it onto the CPU. Or resuming a session but on a different GPU).
        Impl node: we only store session ids in self.open_sessions because the actual Session objects are huge since they contain the ML models, and if that were a local variable to us we would be writing it to disk constantly since we're a SharedObject.
        """
        assert name not in self.active_sessions, "This session is already active"
        assert name not in self.closed_sessions, "This session is already closed"
        assert name in self.open_sessions, "This session does not exist in the list of open sessions"
        # this `config_override` is a partial config dict with just the things being overrided
        path = os.path.join(fs.root,SESS_DIR,Session.get_id(self,name))
        map_location = config_override['device'] if 'device' in config_override else None
        sess = self.fs.load_nosplay(path,map_location=map_location)
        sess.config.update(config_override)
        return sess


class Suggestion:
    def __init__(self, id, hypers, exp):
        self.id = id
        self.hypers = hypers
        self.loss = None
        self.stats = None
    def close(self,loss,stats=None):
        self.loss = loss
        self.stats = stats

def kwargs_of_strargs(strargs,conversion_dict=None):
    """
    `strargs` are just a concise command line format for specifying key/value pairs for string keys and fairly simple values (ie values with no space characters in them). It looks like this:
        At the command line: --hypers activation=tanh batch_size=16 super_cool_bool_option
        This gets converted by argparse to args.hypers = ['activation=tanh' 'batch_size=16' 'super_cool_bool_option'] which is what `strargs` looks like as it enters this function. As you'd expect this function turns a strargs list into a kwargs dict.
    `conversion_dict` is a str->type dict (e.g. 'batch_size':int). It's optional (in which case everything will be left as a str). You only need to include non-str things in the conversion_dict (though you're allowed to include strs as well for clarity).
    """
    res = {}
    for arg in strargs:
        if '=' not in arg:
            res[arg] = True
        else:
            key,val = arg.split('=')
            res[arg] = val

    if conversion_dict is not None:
        for k,conversion in conversion_dict.items():
            if k in res:
                res[k] = conversion(res[k])

class Suggestor:
    def get_sugg(self,exp):
        raise NotImplementedError

class SigoptSuggestor(Suggestor):
    def __init__(self,strargs):
        super().__init__()
        conversion_dict = { # TODO write some way to say 'help' and get all these options
                'token':str,
                'bandwidth':int,
                'budget':int,
                    }
        kwargs = kwargs_of_strargs(strargs,conversion_dict)
    def get_sugg(self,exp):
        pass

class SigoptExperiment(Experiment):
    pass

class SigoptProject(Project):
    pass


def device_of_str(s):
    if s != 'cpu':
        s = int(s)
    return torch.device(s)

def get_arguments():
    parser = argparse.ArgumentParser(description='MPNN for J-Coupling Kaggle competition')
    ## Exclusive arguments (must use exactly one of them)
    parser.add_argument('--new', metavar='expname sessname device', nargs='+',
                        help='Start a new experiment (and a new session) using the specified names on the specified device')
    parser.add_argument('--join', metavar='expname sessname device', nargs='+',
                        help='Join an existing experiment in a new session using the specified names. Provide a device to run on as well. Use --session with this to override values like number of workers, otherwise these values will be pulled from experiment defaults (which are set during --new)')
    parser.add_argument('--resume', metavar='expname sessname [device]',  nargs='+',
                        help='Resume an experiment. Restarts at the last epoch. Optionally provide a device as well, which is just an abbreviated form of adding --session device=#. Use --session to override other values.')
    parser.add_argument('--control', action='store_true', # we set default to None not False so that `exclusive_options` works below (tho actually it prob works anyways since False+False==0).
                        help='Enter the control console for the project')
    ## Non-exclusive arguments
    parser.add_argument('--suggestor', nargs='+',
                        help='Only used with --new. Enter the suggestor type followed by any arguments for the suggestor (no internal "--", use "=" between key and value, and no spaces within a key/value pair). Example: --suggestor sigopt token=REAL bandwidth=4 budget=1000')
    parser.add_argument('--hypers', nargs='+',
                        help='Only used with --new. Enter any hypers you want to overrides the defaults for (no internal "--", use "=" between key and value, and no spaces within a key/value pair). Example: --hypers bins=20 message_len=160')
    parser.add_argument('--session', nargs='+',
                        help='Used with --new, --join, or --resume. Enter any session arguments you want to override the default (for --new) or existing (for --join and --resume, though there is only a limited subset that are allowed in this case) values of (no internal "--", use "=" between key and value, and no spaces within a key/value pair). Example: --session workers=3 no_shuffle=True')
    parser.add_argument('--temp', action='store_true',default=False,
                        help='For use with `new` to create a temp project')
    #parser.add_argument('remainder', nargs=argparse.REMAINDER)

    mg_config = parser.parse_args()

    # Validity assertions
    exclusive_options = ['new','join','resume','control']
    selected_opt = getattr(mg_config,x) is not None for x in exclusive_options] # there should be one True in here at the index in exclusive_options for the option that was selected
    assert sum(selected_opt) == 1, f"You must provide exactly one of the options: {exclusive_options}"
    mg_config.target = exclusive_option[selected_opt.index(True)] # the args for whatever the selected option was
    if mg_config.new:
        mg_config.mode == 'new'
    elif mg_config.join:
        mg_config.mode == 'join'
    elif mg_config.resume:
        mg_config.mode == 'resume'
    elif mg_config.control:
        mg_config.mode == 'control'
    else:
        raise Exception("Unrecognized mode")

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



class Config:
    def __init__(self,dict):
        assert set(dict.keys()) == set(get_default_sess_config().keys()), "Config should only be used for complete configs with all possible keys"
        for k,v in dict.items():
            self[k] = v
    def __getitem__(self,key):
        return getattr(self,key)
    def __setitem__(self,key,val):
        return setattr(self,key,val)
    # same as the dict.update() method
    def update(self,dict):
        for k,v in dict.items():
            assert k in get_default_sess_config().keys()
            self[k] = v
    def clone(self):
        return Config(self.__items__)



# note that if you --join an experiment you use the Experiment's version of default arguments rather than this
def get_default_sess_config():
    # this is a function rather than a global dict so that it can serve as a personal dict for anyone who calls it and they wont be modifying the global version if the modify it.
    return {
        'workers':4,
        'subset':1.,
        'no_shuffle':False,
        'print_net':False,
        'device':None,
        'random_seed':1234,
        'epochs':30,
        }
sess_config_types = {key:type(val) for key,val in get_default_sess_config().items()}

# --session args allowed for --resume --join and --new
allowed_sess_config = {
        'new': default_sess_config.keys(),
        'join': ['workers','print_net','device','epochs'],
        'resume': ['workers','print_net','device','epochs'],
        }


class Session:
    """
    A Session has a particular set of hyperparameters (ie a suggestion) and belongs to an experiment
    It also has a Stats object
    It also has some configuration info like a specific gpu device number, a file to save to, etc

    Sessions must be instantiated with a name, which needs to be unique to the Experiment but can be shared between experiments -- a simple name might be to name it after the GPU it's running on eg '0'. Under the hood your name will be mangled into: sess_id={experiment}_{name}, but as a user you can just refer to it with whatever name you gave it.

    Sessions live at mg/sessions/ and they're splayed files

    Session.save() is called at the very end of each training loop (after curr_epoch has been incremented)
    Session.save('stats') and other splayed saves are also used at various points
    Session.quicksave() saves everything other than the state_dicts

    Because they are accessed so frequently and involve a large amount of data, these are not SharedObjects. We wouldn't gain much from SharedObjects and it would make for lots of extra boilerplate, unreliability, and slowdowns.

    You can add arbitrary fields to the Session object at runtime and they'll get properly saved (you may have to modify the splay() function if torch.save isn't able to save them or you don't want to save them)
    """
    def __init__(self,name,sess_config,expname):
        assert isinstance(exp,Experiment)
        assert isinstance(sess_config,Config)
        self.name = name # a name unique within the experiment but not between experiments
        self.expname = exp.name
        self.id = Session.get_id(exp,name) # an identifier unique within the project
        self.config = sess_config
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.stats = None
        self.sugg = None
        self.hypers = None

    def save(self,attr_str=None):
        """
        Save yourself to disk via syncfs. Optionally only save specific fields using attr_str (see SyncedFS.save)
        """
        pass
    def quicksave(self):
        """
        Same as `save` but excludes state dicts, speeding it up a lot
        """
        pass
    @staticmethod
    def get_id(exp,name):
        """
        Returns a unique identifier within the Project. Note that Session.name alone is not unique.
        """
        return f"{exp.name}_{name}"
    @staticmethod
    def load(exp,name):
        """
        Load a session from disk
        """
        id = Session.get_id(exp,name)
        pass


"""
if config.device == session_dict['raw']['config'].device:
    util.red(f"Warning: device {config.device} is also being used by the experiment you are `--join`ing off of")
"""
"""
if self.stats.curr_epoch >= self.config.epochs:
    self.config.epochs = self.stats.curr_epoch + self.config.epochs
    print(f"Auto-adjusting max epochs to {self.config.epochs}")
print(f"Model will run from epoch {self.stats.curr_epoch}->{self.config.epochs}")
"""
"""
make a 'creating new unsaved model'
"""

    def _load(self,session_dict):
        for key,val in session_dict['raw'].items():
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
        return f"session({body})"

    def save(self):
        """
        Save everything!
        """
        blacklist = ['train_loader','valid_loader','test_loader','conn', 'plt','policy']
        if self.file is None:
            return
        session_dict = {'raw':{}, 'state_dicts':{}}
        for k,v in self.__dict__.items():
            if k in blacklist:
                continue
            if hasattr(v,'state_dict'):
                session_dict['state_dicts'][k] = v.state_dict()
            else:
                session_dict['raw'][k] = v
        util.gray(f"saving raw: {list(session_dict['raw'].keys())}")
        torch.save(session_dict,f"{self.file}")
        torch.save(self.stats,f"{self.file}.stats")
        util.gray(f"[saved to {self.file}]")

    @staticmethod
    def config_override(oldsession_dict, newconfig):
        """
        Overrides certain fields in `oldsession.config` using ones from `newconfig` as specified in `newconfig.override` and `newconfig.override_all`
        Note that in reality we actually copy all fields from `old` into `new` as long as override is not specified. This allows better compatibility with fields added to the program that are present in newconfig but not oldconfig.
        """
        oldconfig = oldsession_dict['raw']['config']
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
        self.type_cast = typecast_of_typestr(self.type)
        self.tunable = tunable if (derive_fn is None) else False # always False if derive_fn provided
        self.default=default
        self.val=default # this will get modified by HypersConfig.assign_parameters() among other things
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
    def modify(self,min=None,max=None,default=None,opts=None):
        """
        Lets you modify the max/min/default for Param, and it'll adjust `self.tunable` accordingly
        You should use this BEFORE embedding Param in HypersConfig so that the list of tunable params is properly updated, along with any other stuff HypersConfig uses.
        """
        if default is not None:
            self.val = default
        if min is not None:
            assert max is not None
            self.tunable = True
            self.min = min
            self.max = max

    def __repr__(self):
        return f"{self.name}={self.val}"

class CommonCase: # since multiline closures aren't pickleable but classes are
    def __init__(self,other_name,name):
        self.other_name = other_name
        self.name = name
    def __call__(self,hypers_config):
        assert self.other_name in hypers.keys, f"Unable to derive `{self.name}` from value `{self.other_name}` because `{self.other_name}` not found in the hypers list"
        assert hypers.keys.index(self.other_name) < hypers.keys.index(self.name),"{self.name} is derived from {self.other_name} but is defined before {self.other_name}! Please move the derived value after the source value"
        return hypers_config[self.other_name].val

class Hypers:
    """
    A set of hyperparameters generated by HypersConfig based on a Suggestion.
    If a hyperparameter is named Fe_3 you can access it in any of these ways:
        hypers.Fe_3
        hypers['Fe_3']
        hypers.bygroup('Fe',3)
    """
    def __init__(self,params):
        for name,param in params.items():
            self[name] = param.val
    def __getitem__(self,key):
        return getattr(self,key)
    def __setitem__(self,key,val):
        setattr(self,key,val)
    def bygroup(self,key,groupno):
        return self[key+'_'+str(layerno)]
    def __repr__(self):
        body = []
        for k,v in self.__dict__.items():
            body.append(f"{k}={v}")
        body = ','.join(body)
        return f"Hypers({body})"



def get_default_params():
    # It is important that this is a function so when you modify the result of it you are modifying a newly created list of newly created parameters.
    return [
        ## Gauss expansion
        Param('gauss_bins',default=10,tunable=False),
        Param('gauss_overlap',default=1,tunable=False),
        Param('gauss_maxdist',default=8,tunable=False),

        ## high level stuff
        Param('lr',min=1e-5,max=1e-3,type='float',default=1e-4),
        Param('batch_size',min=16,max=256,default=64),
        Param('batch_norm',type='bool',default=True),
        Param('activation',type='categorical',opts=['relu','tanh'],default='relu'),
        Param('num_layers',min=1,max=6,default=3),
        Param('hiddens_per_layer',min=1,max=5,default=2),
        Param('dropout',min=.1,max=.5,type='float',default=.2),

        ## MPNN shapes
        Param('Fv_0',default=dataset.natoms,tunable=False),
        Param('Fv',min=25,max=200,default=100),
        Param('Fe_0',derive_fn='gauss_bins'),
        Param('Fe',min=25,max=200,default=100),
        Param('message_len',min=50,max=250,default=150),
        Param('message_hidden',derive_fn='message_len'),
        Param('vupdate_hidden',derive_fn='Fv'),
        Param('eupdate_hidden',derive_fn='Fe'),

        ## Fingerprint shapes
        Param('Fe_fp',default=1,tunable=False),
        Param('hidden_fp',min=25,max=300,default=150),
        ]
param_type_dict = {param.name:param.type_cast for param in get_default_params()}

def typecast_of_typestr(s):
    if s == 'int':
        return int
    if s == 'float':
        return float
    if s == 'bool':
        return bool
    return str

"""
Be sure to call hypers.finalize() before using any hypers if you manually change things around. Finalize() is automatically called for you: 1) at the end of __init__, 2) at the start of get_sigopt_parameters, and 3) at the end of assign_parameters.
IMPORTANT: derived values must come AFTER the values that they are derived from in the hypers list
"""
class HypersConfig:
    def __init__(self,hypers_strargs):
        hypers_kwargs = kwargs_of_strargs(hypers_strargs,param_type_dict)
        params = get_default_params() # get a clean new default param list

        # do modifications to the param list using `hypers_kwargs`
        for param in params:
            if param.name in hypers_kwargs:
                param.modify(default=hypers_kwargs[param.name])

        # these two dicts go from param_name -> Param
        self.params = {param.name:param for param in params}
        self.tunable_params = {param.name:param for param in params if param.tunable}

        # We also embed all the params as attributes in HypersConfig so you can do hypers_config.message_len as the equivalent of hypers_config.params['message_len']
        for param in params:
            assert not hasattr(self,param.name)
            self[param.name] = param

        self.do_derivations()

    def get_hypers(self):
        self.do_derivations() # to be safe
        return Hypers(self.params)

    def do_derivations(self):
        """
        Some Params are derived from others, for example we might set the input number of edge features to be derived from the number of gaussian bins. This function does all the derivations.
        """
        for param in self.params.values():
            if param.derived:
                param.val = param.derive_fn(self)

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
        # This lives in hypers because do_derivations needs to be called at the end
        for name,val in assignments.items():
            if self[name].type == 'bool':
                val = {'True':True,'False':False}[val]
            self[name].val = val
        self.do_derivations()

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
A policy embedded in a Session for deciding when to end early. check_abort takes a Stats object and returns True if it should abort. This gets called at the end of trainloop right after stats.curr_epoch+=1 happens so curr_epoch == whatever the next epoch to be potentially run is.
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









