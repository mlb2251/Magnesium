import os
from os.path import join,isdir,isfile
from os import mkdir,rmdir,makedirs
from time import sleep
import torch

import time
from collections import namedtuple
from contextlib import contextmanager
import util

import pdb
import traceback
import sys


#TODO REWRITE USING THESE PATH OBJECTS https://docs.python.org/3/library/pathlib.html

@contextmanager
def debug(do_debug):
    try:
        yield None
    except Exception as e:
        if do_debug:
            print(''.join(traceback.format_exception(e.__class__,e,e.__traceback__)))
            print(util.format_exception(e,''))
            pdb.post_mortem()
            sys.exit(1)
        else:
            raise e
    finally:
        pass


# absolute path to file, absolute path to lockfile for the file, absolute path to guard file for the lockfile
Paths = namedtuple('Paths','absolute lock guard relative')
LockMsg = namedtuple('LockMsg','pid time name')

EXPIRATION_TIMEOUT = 10 # seconds till a lock expires
GUARD_EXTENSION = '.fslockguard'
LOCK_EXTENSION = '.fslock'

# TODO rewrite Locks as their own class to clean up SyncedFS

"""
Notations:
`fpath` refers to any filepath, relative or absolute it doesn't matter. If a fn takes an fpath you can give it anything and it'll figure it out.
`abspath` is ofc an absolute path as in Paths.absolute
`lockpath` refers to an absolute path to a lock as in Paths.lock
`paths` refers to a Paths namedtuple
Most other paths are absolute.

"""

class SyncedFS:
    """
    A Synchronized filesystem.

    self.root is the location of the root of the filesystem (absolute). `root` provided doesnt have to be absolute tho.
    self.lockeddir is the location of the .fslock folder where the lock based mirror of the filesystem will be kept (inside root)

    When you lock the directory mg/foo/bar/ you first create the file mg/.fslock/foo/bar.lock any anyone who _try_lock()s on anything prefixed with mg/foo/bar will fail. Meanwhile you take locks on every file/folder under mg/foo/bar recursively. Taking foo/bar.lock functions like a writer lock in a rwlock system

    `with locks()` should let you take mult locks, which is careful to avoid deadlocking with other ppl by giving up early. Note that starvation is possible especially if you take a lot of locks in the way, and taking a directory lock will avoid this.

    """
    def __init__(self,root,name,verbose=False):
        # we make these local here so you cant accidently use them outside
        self.name = name # a name that all your locks will be tagged with
        assert ' ' not in self.name
        # root dir ends with '/'
        root = os.path.realpath(root)
        if root[-1] != '/': # IMP to do this after `realpath`
            root += '/'
        self.root = root
        if not isdir(self.root):
            raise ValueError(f"Root directory {root} not found")

        self.lock_time = {}  # lockpath -> [time that it was created]
        self.lock_count = {} # lockpath -> [number of locks you're holding on that path (ie recursive lock() calls)]
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
    def open(self,fpath,*args,metafile=False,**kwargs):
        if self.verbose: print("open() wrapper called")
        if not metafile and self.ismetafile(fpath):
            raise ValueError(f"{fpath} is a path to a metafile. Use metafile=True to confirm that you are trying to open a file used internally by the FS")
        with self.lock(fpath):
            yield open(self.absolute(fpath),*args,**kwargs) # by using yield we let the `open` contextmanager also do its thing

    def remove(self,fpath,metafile=False):
        abspath = self.get_paths(fpath).absolute
        if self.verbose: print("remove() wrapper called")
        if not metafile and self.ismetafile(fpath):
            raise ValueError(f"{fpath} is a path to a metafile. Use metafile=True to confirm that you are trying to remove a file used internally by the FS")
        with self.lock(fpath):
            os.remove(self.absolute(fpath))

    def makedirs(self,fpath): # TODO (nonurgent) modify to use locking
        makedirs(self.absolute(fpath))

    def mkdir(self,fpath):
        abspath = self.get_paths(fpath).absolute
        if self.ismetafile(fpath):
            raise ValueError(f"{fpath} is a path to a metafile, you can't make a directory with this reserved name")
        with self.lock(fpath):
            mkdir(abspath)

    def isdir(self,fpath):
        return os.path.isdir(self.get_paths(fpath).absolute)
    def isfile(self,fpath):
        return os.path.isfile(self.get_paths(fpath).absolute)
    def ismetafile(self,fpath):
        abspath = self.get_paths(fpath).absolute
        return (abspath.endswith(LOCK_EXTENSION) or abspath.endswith(GUARD_EXTENSION))
    def absolute(self,fpath):
        return self.get_paths(fpath).absolute

    def listdir(self,fpath,metafile=False,type=None):
        all_files = os.listdir(self.absolute(fpath))
        if type == 'dir':
            all_files = list(filter(self.isdir, all_files))
        elif type == 'file':
            all_files = list(filter(self.isfile, all_files))
        if metafile: # include metafiles in results
            return all_files
        return list(filter(lambda f: not self.ismetafile(f), all_files)) # strip out metafiles

    def wipe(self,dir=''):
        if dir == '':
            if self.verbose: util.blue('wipe start')
            self.unlock_all()
        for item in self.listdir(dir):
            to_delete = join(self.root,dir,item)
            if self.verbose: util.blue(f"wiping {to_delete}")
            assert to_delete.startswith(self.root)
            assert to_delete.startswith(os.environ['HOME']) # im just really scared of deleting '/' somehow
            if self.isfile(to_delete):
                self.remove(to_delete)
            if self.isdir(to_delete):
                self.wipe(dir=to_delete)
                self.rmdir(to_delete)
        if dir == '':
            self.lock_time = {}
            self.lock_count = {}
            if self.verbose: util.blue('wipe end')

    def unlock_all(self,dir=''):
        for item in self.listdir(dir,metafile=True):
            to_delete = join(self.root,dir,item)
            if self.isdir(to_delete): # recurse on contents of dirs
                self.unlock_all(dir=to_delete)
            if not self.ismetafile(to_delete):
                continue # dont delete anything except metafiles
            assert to_delete.startswith(self.root)
            assert to_delete.startswith(os.environ['HOME']) # im just really scared of deleting '/' somehow
            self.remove(to_delete,metafile=True)

    def rmdir(self,fpath,recursive=False,metafile=False):
        abspath = self.get_paths(fpath).absolute
        if self.verbose: util.red(f"removing {abspath}")
        with self.lock(fpath):
            if recursive:
                for item in self.listdir(abspath):
                    to_delete = join(abspath,item)
                    assert to_delete.startswith(self.root)
                    assert to_delete.startswith(os.environ['HOME']) # im just really scared of deleting '/' somehow
                    if self.isfile(to_delete):
                        self.remove(to_delete,metafile=metafile)
                    if self.isdir(to_delete):
                        self.rmdir(to_delete,recursive=True,metafile=metafile)
            rmdir(abspath)


    def no_locks(self):
        assert len(self.lock_time) == 0, f"{self.lock_time}"

    def load_nosplay(self, fpath, **kwargs):
        # TODO make a load() for splayed stuff
        """
        Does a locked load on a file using torch.load()
        Note that the lock is released after the load is finished, so you should lock fpath yourself if you want to make changes then write them while keeping it locked the whole time.
        """
        abspath = self.get_paths(fpath).absolute
        with self.lock(fpath):
            loaded = torch.load(abspath,**kwargs)
        return loaded

    @contextmanager
    def modify(self,fpath, expiration_check=True, strict=True, autosave=True):
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
    def lock(self,fpath, expiration_check=True):
        """
        Spins until a lock is acquired on a file. `fpath` can be absolute or relative and error checking will be done for you.
        Also checks if a lock is expired and automatically unlocks it if so.
        """
        #if self.verbose: print(f"lock({fpath}) called")
        self.manual_lock(fpath,expiration_check=expiration_check)
        try:
            yield None
        finally:
            self.unlock(fpath)

    def haslock(self,fpath):
        """
        Takes a file path (NOT a lockpath) and checks if we're holding the lock for it
        """
        return self.get_paths(fpath).lock in self.lock_time

    def save(self,obj,fpath,attr_str=None):
        splay = Splay(obj,fpath,self,attr_str=attr_str)
        splay.save()

    def save_nosplay(self,obj,fpath,no_lock=False):
        """
        Does a torch.save on the provided object, using a temp file for safety in case we're interrupted. Doesn't do any splaying. Use `no_lock` to not take a lock (perhaps needed for some internal files that dont have locks). `path_check` is there for the same reason, eg if you wanna forcibly save something with a reserved extension.
        """
        abspath = self.absolute(fpath)
        with (self.lock(fpath) if not no_lock else nocontext()):
            torch.save(obj,f"{abspath}.tmp.{os.getpid()}") # safe since varnames cant have dots anyways
            os.rename(f"{abspath}.tmp.{os.getpid()}",abspath)

    def manual_lock(self, fpath, expiration_check=True):
        """
        Spins until a lock is acquired on a file. `fpath` can be absolute or relative and error checking will be done for you.
        Also checks if a lock is expired (as long as `expiration_check=True`) and automatically unlocks it if so.
        Doesn't return anything.
        *Lock must be freed with fs.unlock() at the end of use.*
        See `lock()` for a contextmanager version of this with unlocks at the end for you, and should be used the vast majority of the time.
        """
        #if self.verbose: print(f"manual_lock({fpath}) called")
        paths = self.get_paths(fpath)
        if self.ismetafile(fpath):
            return # we don't lock lock files and guard extensions but its useful to be able to call lock() on them when doing things like iterating over directories

        tstart = time.time()
        # spinlock with _try_lock() attempts
        while not self._try_lock(paths.lock):
            if expiration_check:
                self._unlock_if_expired(paths)
            if time.time()-tstart > 5:
                print("Been waiting for a lock for over 5 seconds...")

        # If we just locked a directory, we now lock all sub items
        if self.isdir(fpath):
            if self.verbose: util.green("took a directory lock, so now taking locks on all sub items")
            for item in self.listdir(fpath):
                self.manual_lock(join(fpath,item),expiration_check=expiration_check)

    def unlock(self,fpath):
        """
        Unlock a lock that you own, throwing an error if it's already unlocked, or you dont own it.
        See _unlock(ours=False) if you'd like to unlock a lock that you don't own.
        """
        if self.verbose: print(f"unlock({fpath}) called")
        paths = self.get_paths(fpath)
        if self.ismetafile(fpath):
            return # we don't lock lock files and guard extensions but its useful to be able to call lock()/unlock() on them when doing things like iterating over directories

        if paths.lock not in self.lock_time:
            raise LockError("You can't free a lock you didn't take!")

        # grab the guard, which is occupied in rare cases through certain race conditions with _unlock_if_expired etc
        start = time.time()
        while self._try_lock(paths.guard) is False:
            if time.time() - start > 5:
                print("Been waiting for a guard that should almost never be taken for 5 seconds")

        #if self.verbose: print(f"acquired guard lock")
        try:
            self._unlock(paths.lock) ## unlocking it
        finally: # free up the guard no matter what
            #if self.verbose: print(f"released guard lock")
            self._unlock(paths.guard)

        # If we just unlocked a directory, we now unlock all sub items
        if self.isdir(fpath):
            if self.verbose: util.purple("freed a directory lock, so now freeing locks on all sub items")
            for item in self.listdir(fpath):
                self.unlock(join(fpath,item))

    def _get_lockmsg(self,lockpath):
        """
        Gets the message embedded in a lock/guard, which has info like the PID/time/name of whoever created the lock.
        Returns None if the lockfile has been created but the message has not yet been written into it (ie race condition)
        """
        try:
            with open(lockpath,'r') as f:
                text = f.read().split(' ')
                if len(text) < 3:
                    raise LockMsgNotWrittenError # lockfile created but not message in it yet
                pid,lock_time,name = int(text[0]),float(text[1]),text[2]
                lockmsg = LockMsg(pid,lock_time,name)
                return lockmsg
        except FileNotFoundError:
            raise LockNotFoundError
        except OSError as e:
            raise LockError(f"read_lockmsg() failed due to an OSError: {e}")

    def _sanitize(self,fpath):
        """
        Makes sure the provided path doesnt use any reserved names and is somewhere inside self.root, and turns it from whatever it is (aboslute or relative) into a path relative to self.root, with no trailing '/' even if it's a directory.
        """
        fpath = os.path.normpath(fpath) # remove '..' and '///' etc to canonicalize
        if fpath.startswith('/'):
            if not fpath.startswith(self.root):
                raise ValueError(f"File {fpath} is an absolute path not in the root directory {self.root}")
            fpath = fpath[len(self.root):] # get path relative to self.root
        if fpath[-1] == '/': # remove trailing '/' if any
            fpath = fpath[1:]
        return fpath

    def get_paths(self,fpath): # any path -> Paths
        """
        Returns the Path namedtuple given any absolute or relative `fpath`, and does proper checking on `fpath`.
        """
        fpath = self._sanitize(fpath)
        abspath = join(self.root,fpath)
        return Paths(abspath, abspath+LOCK_EXTENSION, abspath+GUARD_EXTENSION, fpath)

    def _try_lock(self,lockpath):
        """
        Makes a single attempt to take a lock, returning True if taken and False if the lock is busy.
        If successful, we add the lock to self.lock_time and increment its lock_count
        Works for both locks and guards
        """
        lock_time = time.time()
        lockmsg = f"{os.getpid()} {lock_time} {self.name}"
        #if self.verbose: print(f"attempting to lock {lockpath}")

        # check if we already have the lock
        if lockpath in self.lock_time:
            self.lock_count[lockpath] += 1
            if self.verbose: print("we already own this lock, incrementing lock_count")
            return True # we let ppl take a lock multiple times if they already own it, incrementing `self.lock_count[lockpath]` each time

        # check for directory locks all along the path that we're locking
        dirs = lockpath[len(self.root):].split('/')
        for i in range(len(dirs)):
            dirlock = ''.join(dirs[:i+1])+LOCK_EXTENSION
            if isfile(dirlock):
                return False # someone locked a directory along the path that we're trying to lock, so we give up

        # try to take hte lock
        try:
            tmpfile = f"{lockpath}.{os.getpid()}"
            with open(lockpath,'x') as f: # open in 'x' mode (O_EXCL and O_CREAT) so it fails if file exists
                f.write(lockmsg)
                if self.verbose: util.green(f"LOCKED {self.get_paths(lockpath).relative}")
            self.lock_time[lockpath] = lock_time
            self.lock_count[lockpath] = 1
            return True # we got the lock!
        except FileExistsError:
            return False # we did not get the lock

    def _unlock(self,lockpath,ours=True):
        """
        When called with `ours`=True, we must own the lock, and it does decrements our lock_count[lockpath] and pop the lock from self.lock_time if the count hits 0.
        When called with `ours`=False, this is unchecked unlocking, it just calls `rm` and throws a LockError if the file doesn't exist.
        """
        #if self.verbose: print(f"_unlock({lockpath})")

        if ours is True:
            assert lockpath in self.lock_time
            self.lock_count[lockpath] -= 1
            if self.lock_count[lockpath] > 0:
                return # we only release a lock once lock_count hits 0

        # remove the lock
        try:
            os.remove(lockpath)
            if self.verbose: util.purple(f"FREED {self.get_paths(lockpath).relative}")
        except OSError as e:
            raise LockError(f"_unlock() failed due to an OSError: {e}")

        if ours is True:
            # cleanup the lock
            time_held = time.time()-self.lock_time[lockpath]
            self.lock_time.pop(lockpath)
            self.lock_count.pop(lockpath)
            if time_held >= EXPIRATION_TIMEOUT:
                raise LockError("You held the lock for too long, that's not allowed!")


    def _unlock_if_expired(self,paths):
        """
        Checks if a lock is expired and safely removes it if so.
        Implementation Note: we do not acquire the guard on a lock unless we already know it's expired, then once we get the guard we recheck the expiration. We don't just take the guard at the start bc this method is called very frequently and we dont want to increase guard traffic unnecessarily given that expiration should not be common anyways.
        """

        # read lock message
        try:
            lockmsg = self._get_lockmsg(paths.lock)
            # see if expired (we do not get the guard yet bc thats unnecessary traffic since locks usually arent expired anyways)
            if time.time()-lockmsg.time < EXPIRATION_TIMEOUT:
                return # it hasn't expired
        except (LockNotFoundError,LockMsgNotWrittenError): # raised by _get_lockmsg()
            return # someone already freed the lock for us


        print(f"[warn] expired lock detected: {paths.lock} {lockmsg}")
        # get the guard
        if not self._try_lock(paths.guard):
            return # someone else has the guard, so we'll let them remove it

        try:
            # now that we have the guard we should check the lock expiration again, in case someone changed it while we were getting the guard (see comment above for why we dont take the guard before checking expiration)
            if time.time()-self._get_lockmsg(paths.lock).time < EXPIRATION_TIMEOUT:
                return

            # unlock the expired lock
            if self.verbose: print(f"[warn] unlocking expired lock")
            if is_alive(lockmsg.pid):
                print(f"The creator of this lock (pid {lockmsg.pid}) is still alive (unless someone took its pid)")
            else:
                print(f"The creator of this lock (pid {lockmsg.pid}) is not an active process")
            self._unlock(paths.lock,ours=False)
        except (LockNotFoundError,LockMsgNotWrittenError): # raised by _get_lockmsg()
            return
        finally:
            self._unlock(paths.guard)


class LockError(Exception): pass
class LockNotFoundError(LockError): pass
class LockMsgNotWrittenError(LockError): pass

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


    Defined attributes you can use:
        .name: the file name at the end of `relpath` or `path`
        .relpath: path relative to `self.fs.root`
        .abspath: absolute path
        .fs: the SyncedFS -- you can use this to mkdir etc, whatever you want.
    """
    RESERVED = ['fs','name','abspath','relpath','load_count', 'attr_dict']
    def __init__(self,relpath,fs,mode,     *args,**kwargs):
        self.fs = fs
        self.relpath = relpath
        if '/' in relpath:
            self.name = relpath[relpath.rindex('/')+1:]
        else:
            self.name = relpath
        self.abspath = fs.absolute(relpath)
        self.load_count = 0 # depth of load recursion. 0 means we're unloaded, >0 means we're loaded. Same idea as fs.lock_count
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

    def new(self):
        if self.fs.verbose: util.yellow('New sharedobject')
        with self.lock():
            self.fs.save_nosplay({},self.abspath) # save an empty dict at our path. This is our attribute dict.

    def exists(self):
        """
        Boolean indicating if new() has been called on this file path yet (ie is there a saved file at self.path?)
        """
        return self.fs.isfile(self.abspath) or self.fs.isdir(self.abspath)

    @contextmanager
    def lock(self,*args,**kwargs):
        """
        Grabs our lock!
        """
        yield self.fs.lock(self.abspath,*args,**kwargs)

    @contextmanager
    def load(self):
        """
        This is an extremely important contextmanager. If you load do any attr mutation operation you probably want to use it. It holds a lock the whole time and loads the whole shared object into local memory for the duration of the contextmanager. Any getattr/setattr calls will use this local version. This contextmanager recurses with no issues. Nobody else can access the shared object as long as you have it loaded, since load() takes a lock().
        Why this is so important:
            If you do sharedobj.mylist.pop() outside of this contextmanager the 'mylist' attr will be loaded from disk, then pop() will be called on it, but it won't be written back to disk.
            If you have it wrapped in a load() then all of the attributes will be loaded into self.attr_dict, then the 'mylist' attr will simply be taken from this local dict, so when pop() is called it will modify the mylist copy in the local dict, and at the very end everything will get flushed back to disk when load() ends.
        """
        with self.lock():
            self.load_count += 1
            if self.fs.verbose: util.yellow(f'load(load_count={self.load_count})')
            if self.load_count == 1: # if this is first load since being unloaded
                if self.fs.verbose: util.green(f'--loaded--')
                self.attr_dict = self.fs.load_nosplay(self.abspath)
            try:
                yield None
            finally:
                self.load_count -= 1
                if self.fs.verbose: util.yellow(f'unload(load_count={self.load_count})')
                if self.load_count == 0: # unload for real
                    self.fs.save_nosplay(self.attr_dict,self.abspath)
                    if self.fs.verbose: util.purple(f'--unloaded--')

    def __getattr__(self,attr):
        if attr in SharedObject.RESERVED:
            super().__getattr__(attr)
        with self.load(): # loads it if not already loaded
            if self.fs.verbose: util.gray(f'getattr {attr}')
            try:
                return self.attr_dict[attr]
            except KeyError:
                pass # if we raised AttributeError in here the error message would look bad by blaming it on the KeyError
            raise AttributeError(str(attr))

    def __setattr__(self,attr,val):
        if attr in SharedObject.RESERVED:
            super().__setattr__(attr,val)
            return
        with self.load(): # loads it if not already loaded
            if self.fs.verbose: util.gray(f'setattr {attr}')
            self.attr_dict[attr] = val

    def __delattr__(self,attr):
        with self.load():
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
        abspath = fs.get_paths(fpath).absolute

        # this is not actually a splay tree lol, i just like the name. I have no idea what a splay tree actually is.
        self.path = abspath
        self.fs = fs
        self.splay_tree = self.build_splay_tree(obj)
    def save(self):
        with self.fs.lock(self.path):
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
            # recursively build_splay_tree() on all the attrs
            attr_dict = { attr:self.build_splay_tree(val) for attr,val in attr_dict.items() }
            return attr_dict
        # normal object with no mgsave
        return Leaf(obj) # to differentiate it from None or dict etc.
    def do_save(self,save_val,save_path):
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
                self.do_save(val,path)
        else:
            raise Exception("Unrecognized type sent to `do_save`")

class Leaf:
    def __init__(self,val):
        self.val=val


def main(mode):
    fs = SyncedFS('mg','test',verbose=True)
    fs.unlock_all()

    # message writing test
    with fs.open('test','w') as f:
        print("START")
        f.write('message1\n')
        f.write('message2')
        print("END")

    # make sure that message was written successfully
    with fs.open('test','r') as f:
        assert f.read() == 'message1\nmessage2'

    # recursive locks test
    with fs.lock('test'):
        with fs.lock('test'):
            with fs.open('test','w') as f:
                f.write('message')


    # make sure that message was written successfully
    with fs.open('test','r') as f:
        assert f.read() == 'message'



    # multiprocess tests
    if mode == 1:
        with fs.open('test','w') as f:
            print("START")
            print('Now launch this process again (preferably multiple copies) within 7 seconds!')
            sleep(7)
            f.write('message')
            print("END")
    elif mode == 2:
        util.green("Parent trying to get lock")
        fs.manual_lock('test2') # take a lock and dont release it
        if self.verbose: util.green("Parent got lock")
        if os.fork() != 0:
            util.green("Parent sleeping with lock")
            sleep(15) # need to sleep a while in the parent otherwise the lock will be released when the child clears the parent since its an old pid
            util.green("Parent exiting")
            exit(0)
        else:
            sleep(1)
            fs = SyncedFS('mg','test',verbose=True) # so we dont inherit fs.locks from parent
            util.green("Child trying to get lock")
            fs.manual_lock('test2') # take lock in child, which should succeed after 10 seconds
            util.green("Child got lock")
            fs.unlock('test2') # cleanup
            util.green("Child released lock")


    # make sure we aren't still holding any locks
    fs.no_locks()

def dirtest():
    util.green("DIRTEST")
    fs = SyncedFS('mg','test',verbose=True)
    fs.wipe()
    # directories

    # populate testdir
    fs.mkdir('testdir')
    with fs.open('testdir/testfile','w') as f:
        f.write('test')
    with fs.open('testdir/testfile2','w') as f:
        f.write('test')

    # populate testdir/subdir
    fs.mkdir('testdir/subdir')
    with fs.open('testdir/subdir/subfile1','w') as f:
        f.write('test')
    with fs.open('testdir/subdir/subfile2','w') as f:
        f.write('test')

    #populate testdir/subdir/subdir
    fs.mkdir('testdir/subdir/subdir')
    with fs.open('testdir/subdir/subdir/ss1','w') as f:
        f.write('test')
    with fs.open('testdir/subdir/subdir/ss2','w') as f:
        f.write('test')

    # lock testdir/subdir
    fs.manual_lock('testdir/subdir')
    # lock testdir/subdir/subdir
    fs.manual_lock('testdir/subdir/subfile1')
    # lock testdir
    fs.manual_lock('testdir')
    fs.unlock('testdir/subdir')
    fs.unlock('testdir')
    fs.unlock('testdir/subdir/subfile1')
    fs.no_locks()

def wipetest():
    fs = SyncedFS('mg','test',verbose=True)
    fs.wipe()
def dirtest2():
    #util.green("Run wipetest before this, then launch this twice")
    dirtest() # set up some dirs/files for us to work with
    if os.fork() == 0:
        fs = SyncedFS('mg','test',verbose=True)
        util.yellow('FIRST try lock')
        fs.manual_lock('testdir/subdir')
        util.yellow('FIRST got lock')
        sleep(1)
        fs.unlock('testdir/subdir')
        util.yellow('FIRST released lock')
        fs.no_locks()
    else:
        fs = SyncedFS('mg','test',verbose=True)
        sleep(.2)
        util.yellow('SECOND try lock')
        fs.manual_lock('testdir/subdir/subdir/ss2') # will have to wait 1 sec
        util.yellow('SECOND got lock')
        fs.unlock('testdir/subdir/subdir/ss2')
        util.yellow('SECOND released lock')
        fs.no_locks()



def sharedobject_test():
    fs = SyncedFS('mg','test',verbose=True)
    obj = SharedObject('mrtest',fs,'any')
    if hasattr(obj,'a'):
        util.blue(obj.a)
    obj.a = 3
    obj.b = {'test':4}
    print(obj.a)


def sharedobject_test2():
    util.blue("Messages in blue should be counting up from 1")

    if os.fork() == 0:
        fs = SyncedFS('mg','test',verbose=True)
        obj = SharedObject('mrtest',fs,'clean') # make new obj
        with obj.load():
            util.blue(1)
            obj.message = "hey there process 2"
        sleep(.1) # give up lock
        with obj.load():
            util.blue(3)
            assert obj.message == "yo whats up process 1"
            assert obj.list == [1,2,3,4,5]
            assert obj.list is obj.samelist
            obj.list.pop()
            print("sending this:",obj.list)

    else:
        fs = SyncedFS('mg','test',verbose=True)
        sleep(.05)
        obj = SharedObject('mrtest',fs,'old') # access existing obj
        assert obj.message == "hey there process 2"
        with obj.load():
            util.blue(2)
            obj.message = "yo whats up process 1"
            obj.list = [1,2,3,4,5]
            obj.samelist = obj.list
        sleep(.1) # give up lock
        print("got this:",obj.list)
        util.blue(4)



@contextmanager
def nocontext():
    try:
        yield None
    finally:
        pass


if __name__ == '__main__':
    if len(sys.argv) > 1:
        mode = int(sys.argv[1])
    else:
        mode = 0
    with debug(True):
        if mode == 3:
            dirtest()
        elif mode == 4:
            sharedobject_test()
        elif mode == 5:
            wipetest()
        elif mode == 6:
            dirtest2()
        elif mode == 7:
            sharedobject_test2()
        else:
            main(mode)



