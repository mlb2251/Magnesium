import os
from os.path import join,isdir,isfile
from os import mkdir,rmdir,makedirs
from time import sleep

import time
from collections import namedtuple
from contextlib import contextmanager

# absolute path to file, absolute path to lockfile for the file, absolute path to guard file for the lockfile
Paths = namedtuple('Paths','absolute lock guard')
LockMsg = namedtuple('LockMsg','pid time name')

LOCKDIR = '.fslock/'
SHAREDDIR = '.sharedobjs/'
GUARD_EXTENSION = '.lockguard'
LOCKED_DIR_LIST = 'lockeddirlist'
EXPIRATION_TIMEOUT = 10 # seconds till a lock expires

# TODO rewrite Locks as their own class to clean up SyncedFS

"""

Notations:
`fpath` refers to any filepath, relative or absolute it doesn't matter. If a fn takes an fpath you can give it anything and it'll figure it out.
`abspath` is ofc an absolute path
Most other paths are absolute.



"""



class SyncedFS:
    def __init__(self,root,name,verbose=False):
        self.name = name # a name that all your locks will be tagged with
        assert ' ' not in self.name
        # root dir ends with '/'
        if root[-1] != '/':
            root += '/'
        self.root = root
        if not isdir(root):
            raise ValueError(f"Root directory {root} not found")

        self.lockdir = join(root,LOCKDIR)
        if not isdir(self.lockdir):
            mkdir(self.lockdir)
        self.lockeddirlist = join(root,LOCKDIR,LOCKED_DIR_LIST)
        self.shareddir = join(ROOT,SHAREDDIR)
        if not isdir(self.shareddir):
            mkdir(self.shareddir)

        self.locks = {}
        self.lock_count = {}
        self.verbose = verbose

        if self.verbose: print("new SyncedFS created")

    """
    Wrapper on `open` with locking. Don't hold a file open for too long or itll expire after EXPIRATION_TIMEOUT
    Use like normal `open`:
    with open(fname) as f:
        old_values = f.read()
        ...
        f.write(new_values)
    """
    @contextmanager
    def open(self,fpath,*args,**kwargs):
        if self.verbose: print("open() wrapper called")
        with self.lock(fpath):
            yield open(self.get_paths(fpath).absolute,*args,**kwargs)

    def get_shared_obj(self,name):
        return SharedObject(join(self.shareddir,name),self)

    def remove(self,fpath):
        if self.verbose: print("remove() wrapper called")
        with self.lock(fpath):
            os.remove(self.get_paths(fpath).absolute)

    def makedirs(self,fpath):
        makedirs(self.get_paths(fpath).absolute)

    def mkdir(self,fpath,locked=False):
        """
        If `locked` is True then in the lockdir this will be a file rather than a directory, and thus there will be no locks within it (since its a file not a directory). Any reads/writes to anything inside this folder will have to take this single masterlock on the directory
        """
        abspath = self.get_paths(fpath).absolute
        mkdir(abspath) # we mkdir first in case it fails
        if locked is True:
            self._add_lockeddir(abspath)

    def rmdir(self,fpath):
        with self.lock(fpath,no_error_on_dir=True):
            rmdir(self.get_paths(fpath).absolute)

    def no_locks(self):
        assert len(self.locks) == 0, f"{self.locks}"

    ## lockeddir stuff
    def _add_lockeddir(self,abspath):
        """
        After calling _add_lockeddir(abspath) any future read/write to files that start with `abspath` will have to take this lock.
        """
        assert _get_parent_lockeddir(abspath) is None, "Can't make a lockeddir inside a lockeddir"
        _sanitize(abspath) # just do basic error checking on it
        if self.verbose: print(f"adding lockeddir {abspath}")
        with open(self.lockeddirlist,'+') as f:
            paths = f.read().split('\n')
            paths.append(abspath)
            f.write('\n'.join(paths))

    def ensure_mkdir_locked(self,fpath):
        """
        Makes the directory at `fpath` a locked directory. If it already exists and is a locked directory then do nothing. If it already exists and isn't locked, throw an error.
        """
        paths = self.get_paths(fpath)
        assert not isfile(paths.abspath)
        if isdir(paths.abspath): # if it's already created assert that it is a locked directory
            assert _get_parent_lockeddir(paths.abspath) == paths.abspath
        else: # not created
            self.mkdir(abspath,locked=True)

    def _remove_lockeddir(self,abspath):
        if self.verbose: print(f"removing lockeddir {abspath}")
        with open(self.lockeddirlist,'+') as f:
            paths = f.read().split('\n')
            paths.remove(abspath)
            f.write('\n'.join(paths))

    def _get_parent_lockeddir(self,abspath):
        """
        Returns None if none of the parents of `abspath` is a locked directory, otherwise returns the locked directory's absolute path
        """
        with open(self.lockeddirlist,'r') as f:
            paths = f.read().split('\n')
        for path in paths:
            if abspath.startswith(path):
                return path
        return None

    def load(self, fpath, expiration_check=True, no_error_on_dir=False, **kwargs):
        # TODO make this load_nosplay and make another load() for splayed stuff
        """
        Does a locked load on a file using torch.load()
        """
        fpath = self.get_paths(fpath).absolute
        with self.fs.lock(fpath):
            loaded = torch.load(fpath,**kwargs)
        return loaded

    @contextmanager
    def modify(self,fpath, expiration_check=True, no_error_on_dir=False, strict=True, autosave=True):
        """
        Contextmanager for modifying an object saved at a path
        """
        fpath = self.get_paths(fpath).absolute
        with self.fs.lock(fpath):
            loaded = torch.load(fpath)
            savefn = SaveFn(fpath,self)
            try:
                if not autosave:
                    yield loaded,savefn
                else:
                    yield loaded
            finally:
                if autosave:
                    savefn(loaded)
                if savefn.saved is False and strict is True:
                    raise Exception("You forgot to call `savefn` after modifying your value! If you didn't want to call it use `modify(strict=False)`")

    @contextmanager
    def lock(self,fpath, expiration_check=True, no_error_on_dir=False):
        """
        Spins until a lock is acquired on a file. `fpath` can be absolute or relative and error checking will be done for you.
        Also checks if a lock is expired and automatically unlocks it if so.
        """
        if self.verbose: print(f"lock({fpath}) called")
        self.manual_lock(fpath,expiration_check=expiration_check, no_error_on_dir=no_error_on_dir)
        try:
            yield None
        finally:
            print("RELEASING")
            self.unlock(fpath,no_error_on_dir=no_error_on_dir)

    def haslock(self,fpath):
        """
        Takes a file path (NOT a lockfile) and checks if we're holding the lock for it
        """
        if self.get_paths(fpath).lock in self.locks:
            return True
        return False
    def save(self,obj,fpath,attr_str=None):
        splay = Splay(obj,fpath,self,attr_str=attr_str)
        splay.save()
    def save_nosplay(self,obj,fpath,no_lock=False,path_check=True):
        """
        Does a torch.save on the provided object, using a temp file for safety in case we're interrupted. Doesn't do any splaying. Use `no_lock` to not take a lock (perhaps needed for some internal files that dont have locks). `path_check` is there for the same reason, eg if you wanna forcibly save something with a reserved extension.
        """
        if path_check:
            fpath = self.get_paths(fpath).absolute

        if no_lock:
            torch.save(obj,f"{self.path}.tmp.{os.getpid()}") # safe since varnames cant have dots anyways
            os.rename(f"{self.path}.tmp.{os.getpid()}",save_path)
            return
        with self.lock(abspath):
            torch.save(obj,f"{self.path}.tmp.{os.getpid()}") # safe since varnames cant have dots anyways
            os.rename(f"{self.path}.tmp.{os.getpid()}",save_path)

    def manual_lock(self, fpath, expiration_check=True, no_error_on_dir=False):
        """
        Spins until a lock is acquired on a file. `fpath` can be absolute or relative and error checking will be done for you.
        Also checks if a lock is expired and automatically unlocks it if so.
        Returns nothing.
        Lock must be freed with .unlock() at the end of use.
        See `lock` for a contextmanager version of this.
        """
        if self.verbose: print(f"manual_lock({fpath}) called")
        paths = self.get_paths(fpath)

        fail_because_dir = isdir(paths.absolute)

        # check if there's a parent locked directory and if so use that lock instead
        parent = self._get_parent_lockeddir(paths.absolute)
        if parent is not None:
            paths = self.get_paths(parent)
            fail_because_dir = False

        if fail_because_dir and no_error_on_dir:
            return
        if fail_because_dir:
            raise LockError(f"You can't take a lock on a directory that isn't a lockeddir or child of a lockeddir")

        tstart = time.time()
        # spinlock
        while not self._try_lock(paths.lock):
            if expiration_check:
                self._unlock_if_expired(paths)
            if time.time()-tstart > 5:
                print("Been waiting for a lock for over 5 seconds...")

    # unlocks a file, throws an error if it was already unlocked (eg someone unlocked it bc of expiration)
    def unlock(self,fpath,no_error_on_dir=False):
        """
        Unlocks a file, throwing an error if we were not the ones who locked it.
        """
        if self.verbose: print(f"unlock({fpath}) called")
        paths = self.get_paths(fpath)

        fail_because_dir = isdir(paths.absolute)

        # check if there's a parent locked directory and if so use that lock instead
        parent = self._get_parent_lockeddir(paths.absolute)
        if parent is not None:
            paths = self.get_paths(parent)
            fail_because_dir = False

        # this is nearly unnecessary in `unlock` except that we need it bc lock() returns early so unlock() must too or else itll try to unlock a dir that was never locked by lock()
        if fail_because_dir and no_error_on_dir:
            return
        if fail_because_dir:
            raise LockError(f"You can't free a lock on a directory that isn't a lockeddir or child of a lockeddir")

        if paths.lock not in self.locks:
            raise LockError("You can't free a lock you didn't take!")
        if not self._try_lock(paths.guard):
            raise LockError("There shouldn't be any competition for the guard on a lock we own, unless we're over the expiration limit which is not okay!")
        # we now have the guard lock
        # now lets make sure the lock is still ours
        if self.verbose: print(f"acquired guard lock")
        try:
            #lockmsg = self._get_lockmsg(paths.lock)
            #if lockmsg.pid != os.getpid():
            #    raise LockError("We're trying to unlock a lock we don't own!")
            self._unlock(paths.lock) ## unlocking it
        finally: # free up the guard no matter what
            if self.verbose: print(f"released guard lock")
            self._unlock(paths.guard)

    ## under the hood methods (all start with '_')

    def _get_lockmsg(self,lockpath):
        try:
            with open(lockpath,'r') as f:
                text = f.read().split(' ')
                assert len(text) == 3
                pid,lock_time,name = float(text[0]),float(text[1]),text[2]
                lockmsg = LockMsg(pid,lock_time,name)
                return lockmsg
        except OSError as e:
            raise LockError(f"read_lockmsg() failed due to an OSError: {e}")

    def _sanitize(self,fpath,mode='none'):
        """
        Does validity checking on the provided path (no invalid names used), and turns it from whatever it is into a path relative to self.root, with no trailing '/'
        """
        fpath = os.path.normpath(fpath) # remove '..' and '///' etc to canonicalize
        if fpath.startswith('/'):
            if not fpath.startswith(self.root):
                raise ValueError(f"File {fpath} is an absolute path not in the root directory {self.root}")
            fpath = fpath[len(self.root):] # get path relative to self.root
        while fpath[-1] == '/': # remove all trailing '/'
            fpath = fpath[1:]
        #if isdir(join(self.root,fpath)):
        #    raise ValueError(f"{fpath} is a directory, but must be a file (or uncreated) for all syncronization operations")
        if fpath.startswith(LOCKDIR):
            raise ValueError(f"{fpath} starts with the lock directory, which is an invalid place for non-lock files")
        if fpath.endswith(GUARD_EXTENSION):
            raise ValueError(f"{fpath} ends with the lock guard extension {GUARD_EXTENSION} which is invalid for non-guard files")
        return fpath

    def get_paths(self,fpath): # any path -> Paths
        """
        Returns the Path namedtuple given any absolute or relative `fpath`, and does proper checking on `fpath`.
        """
        fpath = self._sanitize(fpath)
        return Paths(join(self.root,fpath),join(self.lockdir,fpath),join(self.lockdir,fpath+GUARD_EXTENSION))

    def _try_lock(self,lockpath):
        """
        Makes a single attempt to take a lock, returning True if taken and False if the lock is busy.
        """
        lock_time = time.time()
        lockmsg = f"{os.getpid()} {lock_time} {self.name}"
        if self.verbose: print(f"attempting to lock {lockpath}")

        if lockpath in self.locks: # we let ppl pretend theyre taking a lock multiple times
            self.lock_count[lockpath] += 1
            return True

        try:
            with open(lockpath,'x') as f: # open in 'x' mode (O_EXCL and O_CREAT) so it fails if file exists
                f.write(lockmsg)
                print(f"wrote lock message to {lockpath}")
            self.locks[lockpath] = lock_time
            self.lock_count[lockpath] = 1
            return True
        except FileExistsError:
            return False

    def _unlock(self,path):
        """
        Unchecked unlocking, just calls `rm` and throws a LockError if the file doesn't exist
        This function doesn't require you to own the lock, though it does decrement your lock_count and pop the lock from self.locks.
        """
        if self.verbose: print(f"_unlock({path})")

        if path in self.locks:
            self.lock_count[path] -= 1
            if self.lock_count[path] != 0:
                return # we only release a lock once lock_count hits 0

        # remove the lock
        try:
            os.remove(path)
        except OSError as e:
            raise LockError(f"_unlock() failed due to an OSError: {e}")
        # cleanup the lock
        self.locks.pop(path)
        if time.time()-self.locks[path] >= EXPIRATION_TIMEOUT:
            raise LockError("You held the lock for too long, that's not allowed!")


    def _unlock_if_expired(self,paths):

        """
        Checks if a lock is expired and safely removes it if so.
        Implementation Note: we do not acquire the guard on a lock unless we already know it's expired, then once we get the guard we recheck the expiration. We don't just take the guard at the start bc this method is called very frequently and we dont want to increase guard traffic unnecessarily given that expiration should not be common anyways.
        """
        try:
            lockmsg = self._get_lockmsg(paths.lock)
        except LockError:
            return # someone already freed the lock for us
        if time.time()-lockmsg.time < EXPIRATION_TIMEOUT:
            return # it hasn't expired
        print("[warn] expired lock detected: {paths.lock} {lockmsg}")
        if not self._try_lock(paths.guard):
            return # someone else has the guard, so we'll let them remove it
        try:
            if self.verbose: print(f"[warn] unlocking expired lock")
            if is_alive(lockmsg.pid):
                print("The creator of this lock (pid {lockmsg.pid}) is still alive (unless someone took its pid)")
            else:
                print("The creator of this lock (pid {lockmsg.pid}) is not an active process")
            self._unlock(paths.lock)
        finally:
            self._unlock(paths.guard)
        return

class LockError(Exception): pass

# must be a class since closures can't be pickled
class SaveFn:
    def __init__(self,fpath,fs):
        self.path = fs.get_paths(fpath).absolute
        self.fs = fs
        self.saved = False
    def __call__(self,obj):
        self.fs.save(obj,self.path)
        self.saved = True


# check if a process with pid `pid` is alive
def is_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

class SharedObject:
    """
    A SharedObject lives on the disk in a file located at `self.path`. All attribute accesses are locked disk reads/writes, so no data actually stays in SharedObject.
    When you call SharedObject("/path/to/storage",fs) it will attach itself to whatever object lives at /path/to/storage, or will create a new object there if none exists.
    If there's one thing you should know about shared objects, it's that you should be using the load() contextmanager very frequently. If you have a weird bug where data is disappearing or state isn't being held you probably need to use load().

    `mode` =
        'any' -> dont throw any errors, just load it if it already exists and create it if it doesn't exist.
        'new' -> call .new() using *args **kwargs. Also throw an error if it already exists()
        'old' -> Throw an error if it doesn't already exists()
        'clean' -> call .new() no matter what, even if it already exists (like 'any' but also wipes it clean)
    """
    def __init__(self,path,fs,mode,     *args,**kwargs):
        self.fs = fs
        self.path = fs.get_paths(path).absolute
        self.loaded = False # whether we're in the self.load() contextmanager
        with self.lock():
            if mode == 'any':
                if not self.exists():
                    self.new(*args,**kwargs)
            elif mode == 'old':
                assert self.exists()
            elif mode == 'new':
                assert not self.exists()
                self.new(*args,**kwargs)
            elif mode == 'clean':
                self.new(*args,**kwargs)
            else:
                raise Exception(f"Mode {mode} not recognized in SharedObject __init__")

    @self.lock()
    def new(self):
        self.fs.save({},self.path) # save an empty dict at our path. This is our attribute dict.

    def exists(self):
        """
        Boolean indicating if new() has been called on this file path yet (ie is there a saved file at self.path?)
        """
        return os.path.isfile(self.path) or self.path.isdir(self.path)

    @contextmanager
    def lock(self,*args,**kwargs):
        """
        Grabs our lock!
        """
        yield self.fs.lock(self.path,*args,**kwargs)

    @contextmanager
    def load(self):
        """
        This is an extremely important contextmanager. If you load do any attr mutation operation you probably want to use it. It holds a lock the whole time and loads the whole shared object into local memory for the duration of the contextmanager. Any getattr/setattr calls will use this local version, a
        Why this is so important:
            If you do sharedobj.mylist.pop() outside of this contextmanager the 'mylist' attr will be loaded from disk, then pop() will be called on it, but it won't be written back to disk.
            If you have it wrapped in a load() then all of the attributes will be loaded into self.attr_dict, then the 'mylist' attr will simply be taken from this local dict, so when pop() is called it will modify the mylist copy in the local dict, and at the very end everything will get flushed back to disk when load() ends.
        """
        with self.lock():
            self.attr_dict = self.fs.load_nosplay(self.path)
            self.loaded = True
            try:
                yield None
            finally:
                self.save_nosplay(self.attr_dict,self.path)
                self.loaded = False

    def __getattr__(self,attr):
        if attr in ['fs','path','sticky','attr_dict']:
            super().__getattr__(attr)
        if self.loaded: # use local copy if `loaded`
            return attr_dict[attr]
        with self.load():
            return self.attr_dict[attr]

    def __setattr__(self,attr,val):
        if attr in ['fs','path','sticky','attr_dict']:
            super().__setattr__(attr,val)
            return
        if self.loaded: # use local copy if `loaded`
            return attr_dict[attr] = val
        with self.load():
            self.attr_dict[attr] = val

    @self.load()
    def __delattr__(self,attr):
        self.attr_dict.pop(attr)


#    @contextmanager
#    def modify_attrs(self,*args,**kwargs):
#        """
#        Modify the attr_dict directly
#        """
#        yield self.fs.modify(self.path,*args,**kwargs)
#    @contextmanager
#    def mod(self,*attrs):
#        """
#        Modify one or more attributes of `self`
#        """
#        with self.fs.modify(self.path) as attr_dict:
#            vals = [attr_dict[attr] for attr in attrs]
#            try:
#                yield vals
#            finally:
#                attr_dict[val] = val

class Splay:
    def __init__(self,obj,fpath,fs,attr_str=None):
        """
        Saves `obj` to path `fpath` in file system `fs` as a splayed object:
            If `obj` has a splay_fn() function it will be called. If it returns None the obj will not be saved. if there is no splay_fn() funciton the whole obj will be saved in one file. if there is a splay function that doesnt return None then it should return a list of attr names, and each attr will be saved as a separate file in a folder. This happens recursively if any of the attrs has a splay_fn() function.
        `attr_str`='sub1.sub2' would indicate that only self.save_obj.sub1.sub2 should be saved, so we'd be updating the file or dir fpath/sub1/sub2.
        Calls build_splay_tree to build the tree of things to save and do_save to save it with torch.save()
        """
        if obj is None:
            print("Not saving because `obj` is None which is probably a mistake and we don't wanna overwrite some big file youve been making with this garbage <3...")
            return

        # deal with `attr_str`
        if attr_str is not None:
            for attr in attr_str.split('.'):
                obj = getattr(obj,attr)
                fpath = os.path.join(fpath,attr)
        fpath = fs.get_paths(fpath).absolute

        # this is not actually a splay tree lol, i just like the name. I have no idea what a splay tree actually is.
        self.path = fpath
        self.fs = fs
        self.splay_tree = self.build_splay_tree(obj)
    def save(self):
        with self.fs.lock(self.path)
            self.do_save(self.splay_tree,self.path)
    def build_splay_tree(self,obj):
        """
        Returns return_val:
            - return_val is None: obj should not be saved
            - type(return_val) == Leaf: return_val.val should be saved directly
            - type(return_val) == dict: recursive. keys are attr strs and values are more `return_val`s. Make a directory and recursively save all the items in it.
        """
        if hasattr(obj,'splay_fn'):
            attr_dict = obj.splay_fn() # dir of attr->val
            if attr_dict is None: # obj.mgsave()->None indicates an object should never be saved
                return None
            # recursively _get_save on all the attrs
            attr_dict = { attr:self.build_splay_tree(val) for attr,val in attr_dict.items() }
            return attr_dict
        # normal object with no mgsave
        return Leaf(obj) # to differentiate it from None or dict etc.
    def do_save(self,save_path,save_val):
        """
        Calls torch.save to actually save a save_val tree
        See comment in `_get_save` to see why we do different things for different types of `get_save`
        """
        if save_val is None: # dont save anything that's a `None` unless it's wrapped in a Leaf()
            pass
        elif type(save_val) == Leaf: # leaf
            print("saving a leaf")
            self.fs.save_nosplay(save_val.val,save_path)
        elif type(save_val) == dict: # tree case
            for attr,val in save_val.items():
                path = os.path.join(save_path,attr)
                self.do_save(self,path,val)
        else:
            raise Exception("Unrecognized type sent to `do_save`")


if __name__ == '__main__':
    fs = SyncedFS('mg','test',verbose=True)
    with fs.open('test','w') as f:
        print("START")
        f.write('ayyyy\nlmao')
        sleep(17)
        f.write('ya\nlmao')
        print("END")
    fs.no_locks()



