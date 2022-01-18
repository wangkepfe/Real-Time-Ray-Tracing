#pragma once

#include <list>
#include <optional>
#include <set>
#include <string>
#include <variant>
#include <vector>

#include "linear_math.h"

struct Triangle;

template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

// Types of sub-division
enum class SubD { linear, catmullclark, loop };

class Halfedge_Mesh {
public:
    /*
        For code clarity, we often want to distinguish between
        an integer that encodes an index (an "ordinal" number)
        from an integer that encodes a size (a "cardinal" number).
    */
    using Index = size_t;
    using Size = size_t;

    /*
        A Halfedge_Mesh is comprised of four atomic element types:
        vertices, edges, faces, and halfedges.
    */
    class Vertex;
    class Edge;
    class Face;
    class Halfedge;

    /*
        Rather than using raw pointers to mesh elements, we store references
        as STL::iterators---for convenience, we give shorter names to these
        iterators (e.g., EdgeRef instead of list<Edge>::iterator).
    */
    using VertexRef = std::list<Vertex>::iterator;
    using EdgeRef = std::list<Edge>::iterator;
    using FaceRef = std::list<Face>::iterator;
    using HalfedgeRef = std::list<Halfedge>::iterator;

    /* This is a special kind of reference that can refer to any of the four
       element types. */
    using ElementRef = std::variant<VertexRef, EdgeRef, HalfedgeRef, FaceRef>;

    /*
        We also need "const" iterator types, for situations where a method takes
        a constant reference or pointer to a Halfedge_Mesh.  Since these types are
        used so frequently, we will use "CIter" as a shorthand abbreviation for
        "constant iterator."
    */
    using VertexCRef = std::list<Vertex>::const_iterator;
    using EdgeCRef = std::list<Edge>::const_iterator;
    using FaceCRef = std::list<Face>::const_iterator;
    using HalfedgeCRef = std::list<Halfedge>::const_iterator;
    using ElementCRef = std::variant<VertexCRef, EdgeCRef, HalfedgeCRef, FaceCRef>;

    //////////////////////////////////////////////////////////////////////////////////////////
    // Student Local Operations | student/meshedit.cpp
    //////////////////////////////////////////////////////////////////////////////////////////

    // Note: if you erase elements in these methods, they will not be erased from the
    // element lists until do_erase or validate are called. This is to facilitate checking
    // for dangling references to elements that will be erased.
    // The rest of the codebase will automatically call validate() after each op,
    // but you may need to be aware of this when implementing global ops.
    // Specifically, when you need to collapse an edge in iostropic_remesh() or simplify(),
    // you should call collapse_edge_erase() instead of collapse_edge()

    /*
        Merge all faces incident on a given vertex, returning a
        pointer to the merged face.
    */
    std::optional<FaceRef> erase_vertex(VertexRef v);

    /*
        Merge the two faces on either side of an edge, returning a
        pointer to the merged face.
    */
    std::optional<FaceRef> erase_edge(EdgeRef e);

    /*
        Collapse an edge, returning a pointer to the collapsed vertex
    */
    std::optional<VertexRef> collapse_edge(EdgeRef e);

    /*
        Collapse a face, returning a pointer to the collapsed vertex
    */
    std::optional<VertexRef> collapse_face(FaceRef f);

    /*
        Flip an edge, returning a pointer to the flipped edge
    */
    std::optional<EdgeRef> flip_edge(EdgeRef e);

    /*
        Split an edge, returning a pointer to the inserted midpoint vertex; the
        halfedge of this vertex should refer to one of the edges in the original
        mesh
    */
    std::optional<VertexRef> split_edge(EdgeRef e);

    /*
        Creates a face in place of the vertex, returning a pointer to the new face
    */
    std::optional<FaceRef> bevel_vertex(VertexRef v);

    /*
        Creates a face in place of the edge, returning a pointer to the new face
    */
    std::optional<FaceRef> bevel_edge(EdgeRef e);

    /*
        Insets a face into the given face, returning a pointer to the new center face
    */
    std::optional<FaceRef> bevel_face(FaceRef f);

    /*
        Computes vertex positions for a face that was just created by beveling a vertex,
        but not yet confirmed.
    */
    void bevel_vertex_positions(const std::vector<Float3>& start_positions, FaceRef face,
                                float tangent_offset);

    /*
        Computes vertex positions for a face that was just created by beveling an edge,
        but not yet confirmed.
    */
    void bevel_edge_positions(const std::vector<Float3>& start_positions, FaceRef face,
                              float tangent_offset);

    /*
        Computes vertex positions for a face that was just created by beveling a face,
        but not yet confirmed.
    */
    void bevel_face_positions(const std::vector<Float3>& start_positions, FaceRef face,
                              float tangent_offset, float normal_offset);

    /*
        Collapse an edge, returning a pointer to the collapsed vertex
        ** Also deletes the erased elements **
    */
    std::optional<VertexRef> collapse_edge_erase(EdgeRef e) {
        auto r = collapse_edge(e);
        do_erase();
        return r;
    }

    //////////////////////////////////////////////////////////////////////////////////////////
    // Student Global Operations | student/meshedit.cpp
    //////////////////////////////////////////////////////////////////////////////////////////

    /*
        Splits all non-triangular faces into triangles.
    */
    void triangulate();

    /*
        Compute new vertex positions for a mesh that splits each polygon
        into quads (by inserting a vertex at the face midpoint and each
        of the edge midpoints).  The new vertex positions will be stored
        in the members Vertex::new_pos, Edge::new_pos, and
        Face::new_pos.  The values of the positions are based on
        simple linear interpolation, e.g., the edge midpoints and face
        centroids.
    */
    void linear_subdivide_positions();

    /*
        Compute new vertex positions for a mesh that splits each polygon
        into quads (by inserting a vertex at the face midpoint and each
        of the edge midpoints).  The new vertex positions will be stored
        in the members Vertex::new_pos, Edge::new_pos, and
        Face::new_pos.  The values of the positions are based on
        the Catmull-Clark rules for subdivision.
    */
    void catmullclark_subdivide_positions();

    /*
        Sub-divide each face based on the Loop subdivision rule
    */
    void loop_subdivide();

    /*
        Isotropic remeshing
    */
    bool isotropic_remesh();

    /*
        Mesh simplification
    */
    bool simplify();

    //////////////////////////////////////////////////////////////////////////////////////////
    // End student operations, begin methods students should use
    //////////////////////////////////////////////////////////////////////////////////////////

    class Vertex {
    public:
        // Returns a halfedge incident from the vertex
        HalfedgeRef& halfedge() {
            return _halfedge;
        }
        HalfedgeCRef halfedge() const {
            return _halfedge;
        }

        // Returns whether the vertex is on a boundary loop
        bool on_boundary() const;
        // Returns the number of edges incident from this vertex
        unsigned int degree() const;
        // Computes an area-weighted normal vector at the vertex
        Float3 normal() const;
        // Returns the position of the vertex
        Float3 center() const;
        // Computes the centroid of the loop of the vertex
        Float3 neighborhood_center() const;
        // Returns an id unique to this vertex
        unsigned int id() const {
            return _id;
        }

        // The vertex position
        Float3 pos;

    private:
        Vertex(unsigned int id) : _id(id) {
        }
        Float3 new_pos;
        bool is_new = false;
        unsigned int _id = 0;
        HalfedgeRef _halfedge;
        friend class Halfedge_Mesh;
    };

    class Edge {
    public:
        // Returns one of the two halfedges associated with this edge
        HalfedgeRef& halfedge() {
            return _halfedge;
        }
        HalfedgeCRef halfedge() const {
            return _halfedge;
        }

        // Returns whether this edge is contained in a boundary loop
        bool on_boundary() const;
        // Returns the center point of the edge
        Float3 center() const;
        // Returns the average of the face normals on either side of this edge
        Float3 normal() const;
        // Returns the length of the edge
        float length() const;
        // Returns an id unique to this edge
        unsigned int id() const {
            return _id;
        }

    private:
        Edge(unsigned int id) : _id(id) {
        }
        Float3 new_pos;
        bool is_new = false;
        unsigned int _id = 0;
        HalfedgeRef _halfedge;
        friend class Halfedge_Mesh;
    };

    class Face {
    public:
        // Returns some halfedge contained within this face
        HalfedgeRef& halfedge() {
            return _halfedge;
        }
        HalfedgeCRef halfedge() const {
            return _halfedge;
        }

        // Returns whether this is a boundary face
        bool is_boundary() const {
            return boundary;
        }
        // Returns the centroid of this face
        Float3 center() const;
        // Returns an area weighted face normal
        Float3 normal() const;
        // Returns the number of vertices/edges in this face
        unsigned int degree() const;
        // Returns an id unique to this face
        unsigned int id() const {
            return _id;
        }

    private:
        Face(unsigned int id, bool is_boundary) : _id(id), boundary(is_boundary) {
        }
        Float3 new_pos;
        unsigned int _id = 0;
        HalfedgeRef _halfedge;
        bool boundary = false;
        friend class Halfedge_Mesh;
    };

    class Halfedge {
    public:
        // Retrives the twin halfedge
        HalfedgeRef& twin() {
            return _twin;
        }
        HalfedgeCRef twin() const {
            return _twin;
        }

        // Retrieves the next halfedge
        HalfedgeRef& next() {
            return _next;
        }
        HalfedgeCRef next() const {
            return _next;
        }

        // Retrieves the associated vertex
        VertexRef& vertex() {
            return _vertex;
        }
        VertexCRef vertex() const {
            return _vertex;
        }

        // Retrieves the associated edge
        EdgeRef& edge() {
            return _edge;
        }
        EdgeCRef edge() const {
            return _edge;
        }

        // Retrieves the associated face
        FaceRef& face() {
            return _face;
        }
        FaceCRef face() const {
            return _face;
        }

        // Returns an id unique to this halfedge
        unsigned int id() const {
            return _id;
        }

        // Returns whether this edge is inside a boundary face
        bool is_boundary() const {
            return _face->is_boundary();
        }

        // Convenience function for setting all members of the halfedge
        void set_neighbors(HalfedgeRef next, HalfedgeRef twin, VertexRef vertex, EdgeRef edge,
                           FaceRef face) {
            _next = next;
            _twin = twin;
            _vertex = vertex;
            _edge = edge;
            _face = face;
        }

    private:
        Halfedge(unsigned int id) : _id(id) {
        }
        unsigned int _id = 0;
        HalfedgeRef _twin, _next;
        VertexRef _vertex;
        EdgeRef _edge;
        FaceRef _face;
        friend class Halfedge_Mesh;
    };

    /*
        These methods delete a specified mesh element. One should think very, very carefully
        about exactly when and how to delete mesh elements, since other elements will often still
        point to the element that is being deleted, and accessing a deleted element will cause your
        program to crash (or worse!). A good exercise to think about is: suppose you're
        iterating over a linked list, and want to delete some of the elements as you go. How do you
        do this without causing any problems? For instance, if you delete the current element, will
        you be able to iterate to the next element?  Etc.

        Note: the elements are not actually deleted until validate() is called in order to
       facilitate checking for dangling references.
    */
    void erase(VertexRef v) {
        verased.insert(v);
    }
    void erase(EdgeRef e) {
        eerased.insert(e);
    }
    void erase(FaceRef f) {
        ferased.insert(f);
    }
    void erase(HalfedgeRef h) {
        herased.insert(h);
    }

    /*
        These methods allocate new mesh elements, returning a pointer (i.e., iterator) to the
        new element. (These methods cannot have const versions, because they modify the mesh!)
    */
    HalfedgeRef new_halfedge() {
        return halfedges.insert(halfedges.end(), Halfedge(next_id++));
    }
    VertexRef new_vertex() {
        return vertices.insert(vertices.end(), Vertex(next_id++));
    }
    EdgeRef new_edge() {
        return edges.insert(edges.end(), Edge(next_id++));
    }
    FaceRef new_face(bool boundary = false) {
        return faces.insert(faces.end(), Face(next_id++, boundary));
    }

    /*
        These methods return iterators to the beginning and end of the lists of
        each type of mesh element.  For instance, to iterate over all vertices
        one can write

            for(VertexRef v = vertices_begin(); v != vertices_end(); v++)
            {
                // do something interesting with v
            }

        Note that we have both const and non-const versions of these functions;when
        a mesh is passed as a constant reference, we would instead write

            for(VertexCRef v = ...)

        rather than VertexRef.
    */
    HalfedgeRef halfedges_begin() {
        return halfedges.begin();
    }
    HalfedgeCRef halfedges_begin() const {
        return halfedges.begin();
    }
    HalfedgeRef halfedges_end() {
        return halfedges.end();
    }
    HalfedgeCRef halfedges_end() const {
        return halfedges.end();
    }
    VertexRef vertices_begin() {
        return vertices.begin();
    }
    VertexCRef vertices_begin() const {
        return vertices.begin();
    }
    VertexRef vertices_end() {
        return vertices.end();
    }
    VertexCRef vertices_end() const {
        return vertices.end();
    }
    EdgeRef edges_begin() {
        return edges.begin();
    }
    EdgeCRef edges_begin() const {
        return edges.begin();
    }
    EdgeRef edges_end() {
        return edges.end();
    }
    EdgeCRef edges_end() const {
        return edges.end();
    }
    FaceRef faces_begin() {
        return faces.begin();
    }
    FaceCRef faces_begin() const {
        return faces.begin();
    }
    FaceRef faces_end() {
        return faces.end();
    }
    FaceCRef faces_end() const {
        return faces.end();
    }

    /*
        This return simple statistics about the current mesh.
    */
    Size n_vertices() const {
        return vertices.size();
    };
    Size n_edges() const {
        return edges.size();
    };
    Size n_faces() const {
        return faces.size();
    };
    Size n_halfedges() const {
        return halfedges.size();
    };

    bool has_boundary() const;
    Size n_boundaries() const;

    /// Check if half-edge mesh is valid
    std::optional<std::pair<ElementRef, std::string>> validate();
    std::optional<std::pair<ElementRef, std::string>> warnings();

    //////////////////////////////////////////////////////////////////////////////////////////
    // End methods students should use, begin internal methods - you don't need to use these
    //////////////////////////////////////////////////////////////////////////////////////////

    // Various ways of constructing meshes
    Halfedge_Mesh();
    //Halfedge_Mesh(const GL::Mesh& mesh);
    Halfedge_Mesh(const std::vector<std::vector<Index>>& polygons, const std::vector<Float3>& verts);
    Halfedge_Mesh(const Halfedge_Mesh& src) = delete;
    Halfedge_Mesh(Halfedge_Mesh&& src) = default;
    ~Halfedge_Mesh() = default;

    // Various ways of copying meshes
    void operator=(const Halfedge_Mesh& src) = delete;
    Halfedge_Mesh& operator=(Halfedge_Mesh&& src) = default;
    void copy_to(Halfedge_Mesh& mesh);
    ElementRef copy_to(Halfedge_Mesh& mesh, unsigned int eid);

    /// Clear mesh of all elements.
    void clear();
    /// Creates new sub-divided mesh with provided scheme
    bool subdivide(SubD strategy);
    /// Export to renderable vertex-index mesh. Indexes the mesh.
    //void to_mesh(GL::Mesh& mesh, bool split_faces) const;
    /// Create mesh from polygon list
    std::string from_poly(const std::vector<std::vector<Index>>& polygons,
                          const std::vector<Float3>& verts);
    /// Create mesh from renderable triangle mesh (beware of connectivity, does not de-duplicate
    /// vertices)
    //std::string from_mesh(const GL::Mesh& mesh);

    void to_triangles(std::vector<Triangle>& tri);

    /// WARNING: erased elements stay in the element lists until do_erase()
    /// or validate() are called
    void do_erase();

    void mark_dirty();
    bool flipped() const {
        return flip_orientation;
    }
    void flip() {
        flip_orientation = !flip_orientation;
    };
    bool render_dirty_flag = false;

    Float3 normal_of(ElementRef elem);
    static Float3 center_of(ElementRef elem);
    static unsigned int id_of(ElementRef elem);

private:
    std::list<Vertex> vertices;
    std::list<Edge> edges;
    std::list<Face> faces;
    std::list<Halfedge> halfedges;

    unsigned int next_id;
    bool flip_orientation = false;

    std::set<VertexRef> verased;
    std::set<EdgeRef> eerased;
    std::set<FaceRef> ferased;
    std::set<HalfedgeRef> herased;
};


/*
	Some algorithms need to know how to compare two iterators (std::map)
	Here we just say that one iterator comes before another if the address of the
	object it points to is smaller. (You should not have to worry about this!)
*/
inline bool operator<(const Halfedge_Mesh::HalfedgeRef& i, const Halfedge_Mesh::HalfedgeRef& j) {
	return &*i < &*j;
}
inline bool operator<(const Halfedge_Mesh::VertexRef& i, const Halfedge_Mesh::VertexRef& j) {
	return &*i < &*j;
}
inline bool operator<(const Halfedge_Mesh::EdgeRef& i, const Halfedge_Mesh::EdgeRef& j) {
	return &*i < &*j;
}
inline bool operator<(const Halfedge_Mesh::FaceRef& i, const Halfedge_Mesh::FaceRef& j) {
	return &*i < &*j;
}
inline bool operator<(const Halfedge_Mesh::HalfedgeCRef& i, const Halfedge_Mesh::HalfedgeCRef& j) {
	return &*i < &*j;
}
inline bool operator<(const Halfedge_Mesh::VertexCRef& i, const Halfedge_Mesh::VertexCRef& j) {
	return &*i < &*j;
}
inline bool operator<(const Halfedge_Mesh::EdgeCRef& i, const Halfedge_Mesh::EdgeCRef& j) {
	return &*i < &*j;
}
inline bool operator<(const Halfedge_Mesh::FaceCRef& i, const Halfedge_Mesh::FaceCRef& j) {
	return &*i < &*j;
}

/*
	Some algorithms need to know how to hash references (std::unordered_map)
	Here we simply hash the unique ID of the element.
*/
namespace std {
	template<> struct hash<Halfedge_Mesh::VertexRef> {
		uint64_t operator()(Halfedge_Mesh::VertexRef key) const {
			static const std::hash<unsigned int> h;
			return h(key->id());
		}
	};
	template<> struct hash<Halfedge_Mesh::EdgeRef> {
		uint64_t operator()(Halfedge_Mesh::EdgeRef key) const {
			static const std::hash<unsigned int> h;
			return h(key->id());
		}
	};
	template<> struct hash<Halfedge_Mesh::FaceRef> {
		uint64_t operator()(Halfedge_Mesh::FaceRef key) const {
			static const std::hash<unsigned int> h;
			return h(key->id());
		}
	};
	template<> struct hash<Halfedge_Mesh::HalfedgeRef> {
		uint64_t operator()(Halfedge_Mesh::HalfedgeRef key) const {
			static const std::hash<unsigned int> h;
			return h(key->id());
		}
	};
	template<> struct hash<Halfedge_Mesh::VertexCRef> {
		uint64_t operator()(Halfedge_Mesh::VertexCRef key) const {
			static const std::hash<unsigned int> h;
			return h(key->id());
		}
	};
	template<> struct hash<Halfedge_Mesh::EdgeCRef> {
		uint64_t operator()(Halfedge_Mesh::EdgeCRef key) const {
			static const std::hash<unsigned int> h;
			return h(key->id());
		}
	};
	template<> struct hash<Halfedge_Mesh::FaceCRef> {
		uint64_t operator()(Halfedge_Mesh::FaceCRef key) const {
			static const std::hash<unsigned int> h;
			return h(key->id());
		}
	};
	template<> struct hash<Halfedge_Mesh::HalfedgeCRef> {
		uint64_t operator()(Halfedge_Mesh::HalfedgeCRef key) const {
			static const std::hash<unsigned int> h;
			return h(key->id());
		}
	};
} // namespace std