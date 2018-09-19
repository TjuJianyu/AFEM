from enum import Enum, unique
import collections
from collections import namedtuple
import pandas as pd 
import networkx as nx
import time
from multiprocessing import Pool
import math
import numpy as np
from collections import Iterable
from tqdm import tqdm

@unique
class ValueType(Enum):
    Id = "id"
    LatLong = 'latlong'
    Categorical = "categorical"
    Boolean = 'boolean'
    Ordinal = 'ordinal'
    Numeric = "numeric"
    Text = 'text'
    Index = 'index'
    Time = "time"  
    
def timer(func):
    def wrapper(*args,**kwargs):
        start_time = time.time()
        res = func(*args,**kwargs)
        end_time = time.time()
        print("func %s,timecost:%.3f" % (func.__name__,(end_time - start_time)))
 
        return res
    return wrapper

class BasePath(object):
    """docstring for BasePath"""
    def __init__(self,pathstype,name=None):
        super(BasePath, self).__init__()
        self.name = name 
        self.pathstype = pathstype
        if len(pathstype) > 0:
            self.pathentities=  [p[0] for p in pathstype] +[pathstype[-1][1]] 
        else:
            self.pathentities = []
        self.inversepathstype = self._inversepathstype()

    def getpathname(self):
        return self.name
    def getpathentities(self):
        return self.pathentities
    def getpathstype(self):
        return self.pathstype
    def getinversepathstype(self):
        return self.inversepathstype
    def _inversepathstype(self):
        tmpnewpath = self.pathstype[::-1]
        newpath = []
        for p in tmpnewpath:
            direction = "<-"
            if p[4] == "<-":
                direction == "->"
            newpath.append((p[1],p[0],p[3],p[2],direction))
        return newpath

    def getlastentityid(self):
        return self.pathentities[0]

class Path(BasePath):
    """docstring for Path"""
    def __init__(self,pathstype,df,firstindex,start_time_index,lastindex,last_time_index,name=None,start_part_id=None):
        super(Path, self).__init__(pathstype,name)
        self.pathdetail = df
        self.firstindex = firstindex
        self.start_time_index = start_time_index
        self.lastindex = lastindex
        self.last_time_index = last_time_index
        self.start_part_id = start_part_id

    def getfirstkey(self):
        return self.firstindex

    def getlastkey(self):
        return self.lastindex

    def getstarttimeindex(self):
        return self.start_time_index

    def getlasttimeindex(self):
        return self.last_time_index

    def getpathdetail(self):

        return self.pathdetail,{'firstindex':self.firstindex, 'lastindex':self.lastindex,
         'starttimeindex':self.start_time_index,'lasttimeindex': self.last_time_index}

    def getstartpartname(self):
        return self.start_part_id
       
            
     
        

class EntitySet(object):
    """docstring for EntitySet"""
    def __init__(self, name):
        super(EntitySet, self).__init__()
        self.name = name
        self.entityset = {}
        self.entitygraph = nx.MultiDiGraph()
    def draw(self):
        #self.entitygraph.d
        nx.draw_networkx(self.entitygraph)
        #nx.draw_circular(self.entitygraph)
        
        #fg = self.entitygraph
        #pos = nx.spring_layout(fg)
        #pos = nx.spring_layout(fg)
        #nx.draw_networkx_nodes(fg, pos, node_shape='.', node_size=20)
        #nx.draw_networkx_edges(fg, pos)
        #nx.draw_networkx_labels(fg, pos)
        #nx.draw_networkx_edge_labels(fg, pos, edge_labels=nx.get_edge_attributes(fg, 'parentkey'))
        #plt.show()

    def entity_from_dataframe(self,entity_id, dataframe, index,time_index=None,variable_types=None):

        self.entityset[entity_id] = Entity(entity_id, dataframe, index,time_index,variable_types)
        self.entitygraph.add_node(entity_id)

    def addrelationship(self,entityA,entityB,keyA,keyB):
        self.entitygraph.add_edge(entityA,entityB, parentkey = entityA+"_"+keyA,sonkey=entityB+"_"+keyB)
    
    def search_path(self,targetnode,maxdepth,max_famous_son):
        shortpaths = nx.shortest_path(self.entitygraph)
        #print shortpaths
        if maxdepth == 0:
            return [targetnode]
        paths, pathstype = self._search_path(shortpaths,targetnode,maxdepth,max_famous_son)
        #return paths,pathstype
        #pathstype = self._pathstype(paths)

        return [BasePath(p,'_p%d' % i) for i,p in enumerate(pathstype)]

    def _search_path(self,shortpaths,targetnode,maxdepth,max_famous_son):
        
        def pathstypetransform(path):
            tmppath = []
            for i in range(len(path)-1):
                tmppath.append((path[i],path[i+1],self.entitygraph[path[i]][path[i+1]][0]['parentkey'],
                                    self.entitygraph[path[i]][path[i+1]][0]['sonkey'],'->'))
               

            return tmppath
        
        if maxdepth == 0 or max_famous_son == -1:
            return []
        paths = []
        pathstype = []
        
        for node in self.entitygraph.nodes():
            
            ps = []

            if node not in shortpaths or targetnode not in shortpaths[node]:
                continue
            # check shortpath from node to targetnode <= maxdepth + 1 and it is self
            if len(shortpaths[node][targetnode]) <= maxdepth+1 and len(shortpaths[node][targetnode])!= 1:
                ps = pathstypetransform(shortpaths[node][targetnode])
                pathstype.append(ps)         
                paths.append(shortpaths[node][targetnode])
               
            if len(shortpaths[node][targetnode]) + 1 > maxdepth + 1:
                continue
                



            for son in [ edge[1] for edge in self.entitygraph.out_edges(node)]:
                
                if len(shortpaths[node][targetnode])>=2 and son == shortpaths[node][targetnode][1] and len(self.entitygraph[node][targetnode])<=1:
                    continue

                if max_famous_son - 1 < 0:
                    continue
             
                sonps = (son,node,self.entitygraph[node][son][0]['sonkey'],
                                    self.entitygraph[node][son][0]['parentkey'],'<-')
                
                submaxdepth = maxdepth - len(shortpaths[node][targetnode])+1 - 1
                submaxfamousson = max_famous_son-1
                if node in self.entitygraph and targetnode in self.entitygraph[node] and len(self.entitygraph[node][targetnode]) >1 and son == shortpaths[node][targetnode][1]:

                    sonps = (son,node,self.entitygraph[node][son][1]['sonkey'],self.entitygraph[node][son][1]['parentkey'],'<-')

                
                pathstype.append([sonps]+ps)
                paths.append([son] + shortpaths[node][targetnode])
                sonpaths,sonpathstype = self._search_path(shortpaths,son,submaxdepth, submaxfamousson)
  
                for sonpath in sonpaths:
                    if len(sonpath) >=1:
                        paths.append(sonpath +[son]+ shortpaths[node][targetnode])
                for sonpathstype_sub in sonpathstype:
                    if len(sonpathstype_sub) >=1:
                        pathstype.append( sonpathstype_sub+[sonps]+ps)
     
        return paths,pathstype
    

    def _pathstype(self,paths):
        pathstype = []
        for path in paths:
            subpathstype = []
            for i in range(len(path)-1):
                try:
                
                    subpathstype.append((path[i],path[i+1],self.entitygraph[path[i]][path[i+1]]['parentkey'],
                     self.entitygraph[path[i]][path[i+1]]['sonkey'],'->'))
                except:
                    subpathstype.append((path[i],path[i+1],self.entitygraph[path[i+1]][path[i]]['sonkey'],
                     self.entitygraph[path[i+1]][path[i]]['parentkey'],'<-'))
            pathstype.append(subpathstype)
        return pathstype

    def collectiontransform(self,path,target):
        pass
    def getentity(self,entityid):
        try:
            return self.entityset[entityid]
        except:
            raise Exception("id '%s' not found" % str(entityid))

class Entity(object):
    """docstring for Entity"""
    def __init__(self, entity_id, dataframe, index,time_index=None,variable_types=None):
        super(Entity, self).__init__()
        self.name = entity_id
        self.dataframe = dataframe.add_prefix(self.name+"_")
        
        self.variable_types = variable_types
        self.index = self.name+"_"+index

        self.dataframe.index = self.dataframe[self.index]
        
        if time_index is None:
            self.time_index = time_index
        else:
            self.time_index = self.name+"_"+time_index 
            if not pd.api.types.is_datetime64_any_dtype(self.dataframe[self.time_index]):
                self.dataframe[self.time_index] = pd.to_datetime(self.dataframe[self.time_index])
        self.variable_types = variable_types
        self.variable_types[index] = ValueType.Index
        if time_index is not None:
            self.variable_types[time_index] = ValueType.Time
    
    def getcolumns(self,columns):
        
        if isinstance(columns,Iterable):
            return self.dataframe[columns]
        else:   
            return self.dataframe[[columns]]


    def getfeattype(self,featname):
        ftype=None
        

        ftype = self.variable_types[featname[len(self.name+"_"):]]

        return ftype
    def getfeatname(self):
        return [ self.name+"_"+v for v in self.variable_types.keys()]
    
    def merge(self,features,path,how='right'):

        df,info = path.getpathdetail()

        merged = self.dataframe[[self.index]+features].rename({self.index:info['lastindex']},axis=1).\
        merge(df,left_on=info['lastindex'],right_on=info['lastindex'],how=how)

        return merged, info



class Function(object):
    """docstring for Function"""
    def __init__(self, arg):
        super(Function, self).__init__()
        self.arg = arg
        

class Generator(object):
    def __init__(self,es):
        super(Generator,self).__init__()
        self.es = es
        self.paths = {}
        self.paths_feature = {}

    def reload_data(self,es):
        self.es = es
    # TODO: target features do not have any meaning until now

    def layer(self,path,start_part=None,start_part_id=None):
        
        #import collections

        #if isinstance(path, collections.Iterable):
        #    return self.layers(path,start_part,start_part_id)
        
        hashpstype = hash(str(path.getpathstype()))
        if hashpstype in self.paths and start_part_id == self.paths[hashpstype][0]:
     
            return self.paths[hashpstype][1]
        else:
            newpath = self._layer(path,start_part,start_part_id)
            self.paths[hashpstype] =[]
            self.paths[hashpstype].append(start_part_id)
            self.paths[hashpstype].append( newpath )
            return newpath
    
    def layers(self,paths,start_part=None,start_part_id=None):
        assert len(paths) > 1 

        
        dataframe,info = self.layer(paths[0],start_part,start_part_id).getpathdetail()

        for p in range(1,len(paths)):
            nextdf,nextinfo = self.layer(paths[p],start_part,start_part_id).getpathdetail()

            renamedict = {}
            for key in info:
                if nextinfo[key] is not info[key]:
                    renamedict[nextinfo[key]] = info[key]
            if len(renamedict) != 0:  
                
                dataframe = dataframe.merge(nextdf.rename(renamedict)[[ info['firstindex'],info['lastindex'] ]],
                                            on=[info['firstindex'],info['lastindex']],
                                            how='inner')
            else:
                dataframe = dataframe.merge(nextdf[[info['firstindex'],info['lastindex']]],
                                            on=[info['firstindex'],info['lastindex']],
                                            how='inner')
                
     
        return Path(paths[0].getpathstype(),dataframe,info['firstindex'],
                    info['starttimeindex'],info['lastindex'],info['lasttimeindex'],paths[0].getpathname(),start_part_id)
                                            
    
    
        
    def defaultfunc(self,path):
        assert len(path.getpathentities()) > 1
        targetentity = self.es.getentity(path.getpathentities()[0] )
        features = targetentity.getfeatname()

        featurefuncmap = {}
        for feature in features:
            featuretype =  targetentity.getfeattype(feature)

            if featuretype == ValueType.Id:
                featurefuncmap[feature] = ['count','nunique']
                
            elif featuretype == ValueType.Categorical:
                featurefuncmap[feature] = ['count','nunique']
            
            elif featuretype == ValueType.Boolean:
                featurefuncmap[feature] = ['sum','mean']
            
            elif featuretype == ValueType.Numeric:
                featurefuncmap[feature] = ['sum','mean','min','max','count']
        
        return featurefuncmap
    

    
    def _layer(self,path,start_part=None,start_part_id=None):
        es = self.es
        pathstype = path.getinversepathstype()
        dataframe = pd.DataFrame()
        leftremain = []
        rightremain = []
        firstkey = None
        start_time_index = None
        lastkey = None
        last_time_index= None
        
        for i in range(len(pathstype)):
            
            edge = pathstype[i]
            left,right,leftkey,rightkey,etype = edge[0],edge[1],edge[2],edge[3],edge[4]
            leftcolumns = set(["%spid%d"% (name,i) for name in set([leftkey,es.entityset[left].index,es.entityset[left].time_index]) - set([None])])
            if i != 0:
                leftcolumns.add(firstkey)
                leftcolumns.add(start_time_index)
                leftcolumns = leftcolumns - set([None])
            else:
                firstkey = es.entityset[left].index + "pid0"
                
                start_time_index = es.entityset[left].time_index
                if start_time_index is not None:
                    start_time_index += "pid0"
            if i == len(pathstype)-1:
                lastkey = es.entityset[right].index + "pid%d" % (i+1)
                last_time_index = es.entityset[right].time_index
                if last_time_index is not None:
                    last_time_index += ("pid%d" % (i+1))
            leftremain.append( list(leftcolumns) )
            
            
            
            rightcolumns = set([es.getentity(right).index, rightkey,es.getentity(right).time_index])-set([None])
            if i < len(pathstype)-1:
                rightcolumns.add(pathstype[i+1][2])
            rightremain.append(list(rightcolumns))
            
        #for i in range(len(pathstype)):
        #    print leftremain[i]
        #    print rightremain[i]
        isleftdirection = False
        print start_part.columns
        for i in range(len(pathstype)):
            edge = pathstype[i]
            left,right,leftkey,rightkey,etype = edge[0],edge[1],edge[2],edge[3],edge[4]
            if isleftdirection == False and etype == "<-":
                isleftdirection =True
            if i == 0 and start_part is not None:
   
                dataframe = start_part[list(set(start_part.columns))].add_suffix("pid%d" % i)[leftremain[i]]
            elif i== 0 and start_part is None:
                print left
                dataframe = es.entityset[left].dataframe.add_suffix("pid%d" % i)[leftremain[i]]

            a = dataframe[list(set(leftremain[i]))].head()
            b = es.entityset[right].dataframe[list(set(rightremain[i]))].add_suffix("pid%d" % (i+1)).head()
   
            
            dataframe = dataframe[leftremain[i]].merge(es.entityset[right].dataframe[rightremain[i]].add_suffix("pid%d" % (i+1)),left_on="%spid%d" %(leftkey, i),right_on="%spid%d" % (rightkey , i+1),how='left')
            if etype=="->" and isleftdirection==True:
                dataframe = dataframe.drop_duplicates(subset=[firstkey,"%spid%d" % (rightkey,i+1)])

        remain =list(set( [firstkey,start_time_index,lastkey,last_time_index]) - set([None]))

        return Path(path.getpathstype(),dataframe[remain],firstkey,
                    start_time_index,lastkey,last_time_index,path.getpathname(),start_part_id)

        
    def pathfilter(self,path,function,start_part=None,start_part_id=None):
        
        
        hashpstype = hash(str(path.getpathstype()))
        if hashpstype not in self.paths_feature or path.getstartpartname() != self.paths_feature[hashpstype][0]:
            merged, info = self.es.getentity(path.getlastentityid()).merge(function.keys(),path)

            self.paths_feature[hashpstype]=(path.getstartpartname(),merged,info)
        else:
            _,merged, info = self.paths_feature[hashpstype]
    

    def aggregate(self,path,function,iftimeuse = True, winsize='all',lagsize='last'):

        def step(lagsize):
                count = float(lagsize[0:-1])
                unitname = lagsize[-1]
                
                unit = 0
                if unitname =='h':
                    unit = 3600000000000
                elif unitname == 'd':
                    unit = 86400000000000
                elif unitname == 'w':
                    unit = 604800000000000
                else:
                    raise Exception("unknow type: %s" % unitname)
                return int(count*unit) 

        hashpstype = hash(str(path.getpathstype()))
        if hashpstype not in self.paths_feature or path.getstartpartname() != self.paths_feature[hashpstype][0]:
            feature_cols = []
            for key in function:
                if isinstance(key,tuple):
                    feature_cols.extend(key)
                else:
                    feature_cols.append(key)
            merged, info = self.es.getentity(path.getlastentityid()).merge(feature_cols,path)

            self.paths_feature[hashpstype]=(path.getstartpartname(),merged,info)
        else:
            _,merged, info = self.paths_feature[hashpstype]

        tmpmerged = merged 
 
        if iftimeuse and path.getstarttimeindex() is not None and path.getlasttimeindex() is not None:
            lag = (merged[path.getstarttimeindex()]-merged[path.getlasttimeindex()]).astype(int)
    
            
            
            if winsize == 'all' and lagsize == 'last':
                tmpmerged = merged[lag>0]

            elif winsize == 'all'and lagsize != 'last':               
                tmpmerged = merged[lag>=step(lagsize)]

            elif winsize != 'all' and lagsize == 'last':
                if step(winsize) > 0:
                    tmpmerged = merged[(lag>0) & lag<step(winsize)]
                else:
                    tmpmerged = merged[(lag<0) & lag>step(winsize)]
      
            else:
                #print 'a'
                if step(winsize) > 0:
 
                    tmpmerged = merged[(lag>=step(lagsize)) & (lag <step(winsize)+step(lagsize))]

                else:
                    tmpmerged = merged[(lag<=step(lagsize)) & (lag >step(winsize)+step(lagsize))]
   
        for feat in function:
            if isinstance(feat, tuple) and len(feat)==2:
                tmpmerged['x(%s)' % (str(feat))] = tmpmerged[feat[0]] * tmpmerged[feat[1]]
                tmpmerged['/(%s)' % (str(feat))] = tmpmerged[feat[0]] / tmpmerged[feat[1]]
                

        
        groups = tmpmerged.groupby(path.getfirstkey())
        featingroups = set(tmpmerged.columns)-set([path.getfirstkey()])

        newfeats = []
        newfeats_name = []

        
        
        for feat in function:
            
            if feat not in featingroups and len(set(feat) - set(featingroups))>0:
            #if feat not in featingroups:
                continue
            
            print feat
            funcs = function[feat]
            for func in funcs:

                if callable(func):
                    if isinstance(feat,tuple):
                        newfeats.append(getattr(groups['x(%s)' % (str(feat))],func)())
                        newfeats.append(getattr(groups['/(%s)' % (str(feat))],func)())
                          

                    else:
                        newfeats_name.append("%s(%s,%s)" % (func, 'x(%s)' % (str(feat)),path.getpathname()) )
                        newfeats_name.append("%s(%s,%s)" % (func, '/(%s)' % (str(feat)),path.getpathname()) )
                        newfeats.append(groups[feat].apply(func))
                        newfeats_name.append("%s(%s,%s)" % (func.__name__, feat,path.getpathname()))
                    
                elif isinstance(func,str):
                    try:
                        if isinstance(feat,tuple):
                            newfeats.append(getattr(groups['x(%s)' % (str(feat))],func)())
                            newfeats.append(getattr(groups['/(%s)' % (str(feat))],func)())
                            
                            newfeats_name.append("%s(%s,%s)" % (func, 'x(%s)' % (str(feat)),path.getpathname()) )
                            newfeats_name.append("%s(%s,%s)" % (func, '/(%s)' % (str(feat)),path.getpathname()) )
                        else:
                            newfeats.append(getattr(groups[feat],func)())
                      
                            newfeats_name.append("%s(%s,%s)" % (func, feat,path.getpathname()) )
                    except AttributeError:
                        raise NotImplementedError("Class %s does not implement %s" % (groups.__class__.__name__, func))
        newfeats = pd.concat(newfeats, axis=1)

        newfeats.columns = newfeats_name
        #newfeats.rename_axis(path.getfirstkey(),inplace=True)
        return newfeats 


    #compute_series [{'path':,"function":,'iftimeuse':,'winsize':,"lagsize":}]
    def add_compute_series(self,compute_series,start_part=None):
        layer(cs['path'],start_part)
        for cs in compute_series:
            path = layer(cs['path'])
            newfeats = aggregate(path,cs['function'],cs['iftimeuse'],cs['winsize'],cs['lagsize'])
            newfeats.merge(path)


    def pathcompute(self,cs,ngroups='auto',njobs=1):
        es = self.es
        paths = cs['path']
        if not isinstance(cs['path'],collections.Iterable):
            paths = [paths]
        
        selected = set()
        for path in paths:
            pathdetail  = path.getpathstype()

            selected.add(pathdetail[-1][3])
        
        selected.add(es.getentity(pathdetail[-1][1]).index)
        selected = list(selected)

        
        starttime = es.getentity(pathdetail[-1][1]).time_index
        if starttime is not None:
            selected.append(starttime)

        ISDROPED = False
        start = es.getentity(pathdetail[-1][1]).getcolumns(selected)
        lenstart = len(start)
        if pathdetail[-1][4] == "<-":
            start = es.getentity(pathdetail[-1][1]).getcolumns(selected).drop_duplicates(subset=selected[1:])
        if len(start) != lenstart:
            ISDROPED = True
            lenstart = len(start)
        if ngroups =='auto':
            ngroups = min(100,len(start))
 
        groups = range(ngroups) * int(math.ceil(len(start) *1.0/ngroups))
        np.random.shuffle(groups)
        groups = groups[:len(start)]
        newfeats = []
        dfgroups = []
        t = time.time()
        for k,g in start.groupby(groups):
            dfgroups.append(g)
        lenchunk = int(ngroups * 1.0 / njobs)

        params = [dfgroups[i:i+lenchunk] for i in xrange(0,len(dfgroups),lenchunk)]
        params[-1].extend(dfgroups[i+lenchunk:])
        
        
        params = [(p,cs['path'],cs["function"],cs['iftimeuse'],cs['winsize'],cs['lagsize']) for p in params]

        result = None
        if njobs == 1:
            result = self.collect_agg(params[0])
  
        else:
            p = Pool(njobs)
            result = p.map(collect_agg,params)
            result = pd.concat(result)

        if not ISDROPED:
            return result

        mergedres = es.getentity(pathdetail[-1][1]).getcolumns(selected[1:]).reset_index(drop=False).\
        merge(result.join(start[selected[1:]],how='left'),on=selected[1:],\
              how='left').drop(selected[1:],axis=1).set_index(es.getentity(pathdetail[-1][1]).index)

        return mergedres

    def collect_agg(self,inputs):

        group,path,function,iftimeuse,winsize,lagsize = inputs
        newfeats = [] 
        for i in tqdm(xrange(len(group))):
            g = group[i]

            newpath = self.layer(path,start_part=g,start_part_id=i)  
            
            detail,name = newpath.getpathdetail()

            
            newfeats.append(self.aggregate(newpath,function,iftimeuse=iftimeuse,winsize=winsize,lagsize=lagsize))
        if len(newfeats)==1:

            return newfeats[0]
        return pd.concat(newfeats)

    




    def layer_sequencal_agg(self,path,es,ngroups = None,njobs=1):
        if ngroups is None:
            dataframe,firstkey,start_time_index,featurenames,last_time_index =layer(inverse(pathstype),es,start_part = None)  
            newfeats = aggregation(dataframe,firstkey,start_time_index,featurenames,last_time_index,pathstype)
            return newfeats
        else:
            start_entity = inverse(pathstype)[0][0]
            groups = range(ngroups) * int(math.ceil(len(es.entityset[start_entity].dataframe) *1.0/ngroups))
            np.random.shuffle(groups)
            groups = groups[:len(es.entityset[start_entity].dataframe)]
            
            newfeats = []
            dfgroups = []
            t = time.time()
            for k,g in es.entityset[start_entity].dataframe.groupby(groups):
                dfgroups.append(g)
            lenchunk = int(ngroups * 1.0 / njobs)

            params = [dfgroups[i:i+lenchunk] for i in xrange(0,len(dfgroups),lenchunk)]
            params[-1].extend(dfgroups[i+lenchunk:] )
            params = [(p,pathstype,es) for p in params]

            if njobs == 1:
                result = collect_agg(params[0])
                return [result]
            else:
                p = Pool(njobs)
                result = p.map(collect_agg,params)
                return result
        
            
    #         timelast = pd.DataFrame((tmpdataframe.groupby(firstkey)[start_time_index].last()-
    #                     tmpdataframe.groupby(firstkey)[last_time_index].max()).astype(int),
    #                                 columns = ['timetolast(%s_%s)' % (start_time_index,last_time_index)])
   
    

        
    def transform(self,path,featurenames,function):
        pass
        
    
    def singlepathcompunation(self,pathstype,targetfeatures,functionset):
        if not checkorder(functionset):
            raise Exception("function set input exception")
        for layerfunc in functionset:
            start = layerfunc[0][0]
            end = layerfunc[0][1]
            subpathstype = pathstype[start:end]
            dataframe,firstkey,start_time_index,featurenames,last_time_index = self.mergelayer(subpathstype,targetfeatures)

    def pathcompunation(self,pathsfunc):
        
        for pf in pathsfunc:
            path = pf[0]
            first_time = pf[1]
            last_time = pf[2]
            features = pf[3]
            func = pf[4]
            asname = pf[5]
            dataframe,firstkey,start_time_index,featurenames,last_time_index = self.mergelayer(subpathstype,targetfeatures)
            time_index = []
            if first_time is not None:
                time_index.append(first_time)
            if last_time is not None:
                time_index.append(last_time)
            select_features = [features] + [firstkey] + [time_index]
            res = func(dataframe[select_features])


