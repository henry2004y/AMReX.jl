#include <jlcxx/jlcxx.hpp>
#include <jlcxx/stl.hpp>

#include <AMReX.H>
#include <AMReX_AmrCore.H>
#include <AMReX_AmrParticles.H>
#include <AMReX_BCRec.H>
#include <AMReX_BCUtil.H>
#include <AMReX_BC_TYPES.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_Geometry.H>
#include <AMReX_Interpolater.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Particle.H>
#include <AMReX_ParticleTile.H>
#include <AMReX_PhysBCFunct.H>
#include <AMReX_PlotFileUtil.H>
#ifdef AMREX_USE_HDF5
#include <AMReX_PlotFileUtilHDF5.H>
#endif
#include <AMReX_RealBox.H>
#include <AMReX_TagBox.H>
#include <AMReX_Utility.H>
#include <AMReX_VisMF.H>
#include <AMReX_YAFluxRegister.H>
#include <AMReX_EdgeFluxRegister.H>
#include <AMReX_MLMG.H>
#include <AMReX_MLPoisson.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLNodeLaplacian.H>
#include <AMReX_iMultiFab.H>

#include <algorithm>
#include <csignal>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

template <class FAB>
class PhysBCFunctNoOpFab {
public:
    void operator()(amrex::FabArray<FAB>& /*mf*/, int /*dcomp*/, int /*ncomp*/,
                    amrex::IntVect const& /*nghost*/, amrex::Real /*time*/,
                    int /*bccomp*/) {}
};

struct ExtDirFillOp {
    double val_lo[3] = {0.0, 0.0, 0.0};
    double val_hi[3] = {0.0, 0.0, 0.0};

    AMREX_GPU_HOST_DEVICE
    void operator()(const amrex::IntVect& iv, amrex::Array4<amrex::Real> const& dest,
                    int dcomp, int numcomp, amrex::GeometryData const& geom,
                    amrex::Real /*time*/, const amrex::BCRec* bcr,
                    int bcomp, int /*orig_comp*/) const noexcept {
        const auto& dom = geom.Domain();
        const int* dlo = dom.loVect();
        const int* dhi = dom.hiVect();
        for (int n = 0; n < numcomp; ++n) {
            const amrex::BCRec& bc = bcr[bcomp + n];
            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                if (iv[dir] < dlo[dir] &&
                    (bc.lo(dir) == amrex::BCType::ext_dir ||
                     bc.lo(dir) == amrex::BCType::ext_dir_cc)) {
                    dest(iv, dcomp + n) = val_lo[dir];
                } else if (iv[dir] > dhi[dir] &&
                           (bc.hi(dir) == amrex::BCType::ext_dir ||
                            bc.hi(dir) == amrex::BCType::ext_dir_cc)) {
                    dest(iv, dcomp + n) = val_hi[dir];
                }
            }
        }
    }
};

std::vector<std::string> split_varnames(const std::string& joined, int ncomp) {
    std::vector<std::string> res;
    if (!joined.empty()) {
        std::stringstream ss(joined);
        std::string item;
        while (std::getline(ss, item, ',')) {
            res.push_back(item);
        }
    }
    while (static_cast<int>(res.size()) < ncomp) {
        res.push_back("var_" + std::to_string(res.size() + 1));
    }
    return res;
}

amrex::Interpolater* get_interpolater(int interp_type) {
    switch (interp_type) {
        case 0: return &amrex::cell_cons_interp;
        case 1: return &amrex::node_bilinear_interp;
        case 2: return &amrex::pc_interp;
        case 3: return &amrex::quadratic_interp;
        default: return &amrex::cell_cons_interp;
    }
}

} // anonymous namespace

// ============================================================================
// Wrapper classes
// ============================================================================

struct JlBox {
    amrex::Box box;

    JlBox() = default;
    explicit JlBox(const amrex::Box& b) : box(b) {}
    JlBox(int lox, int loy, int loz, int hix, int hiy, int hiz)
        : box(amrex::IntVect(lox, loy, loz), amrex::IntVect(hix, hiy, hiz)) {}
    JlBox(int lox, int loy, int loz, int hix, int hiy, int hiz,
          int tx, int ty, int tz)
        : box(amrex::IntVect(lox, loy, loz), amrex::IntVect(hix, hiy, hiz),
              amrex::IntVect(tx, ty, tz)) {}

    int lo(int dir) const { return box.smallEnd(dir); }
    int hi(int dir) const { return box.bigEnd(dir); }
    int length(int dir) const { return box.length(dir); }
    bool is_nodal(int dir) const { return box.type(dir) == amrex::IndexType::NODE; }
    int64_t num_pts() const { return box.numPts(); }
    bool is_empty() const { return !box.ok(); }
    bool contains(int i, int j, int k) const {
        return box.contains(amrex::IntVect(i, j, k));
    }
    JlBox surrounding_nodes() const {
        return JlBox(amrex::surroundingNodes(box));
    }
    JlBox enclosed_cells() const {
        return JlBox(amrex::enclosedCells(box));
    }
    JlBox grow(int n) const {
        amrex::Box b = box;
        b.grow(n);
        return JlBox(b);
    }
    JlBox intersect(const JlBox& other) const {
        return JlBox(box & other.box);
    }
};

struct JlRealBox {
    amrex::RealBox rbox;

    JlRealBox() = default;
    explicit JlRealBox(const amrex::RealBox& rb) : rbox(rb) {}
    JlRealBox(double xlo, double ylo, double zlo, double xhi, double yhi, double zhi)
        : rbox({xlo, ylo, zlo}, {xhi, yhi, zhi}) {}

    double lo(int dir) const { return rbox.lo(dir); }
    double hi(int dir) const { return rbox.hi(dir); }
    double length(int dir) const { return rbox.length(dir); }
    bool contains(double x, double y, double z) const {
        amrex::Real pt[3] = {x, y, z};
        return rbox.contains(pt);
    }
};

struct JlGeometry {
    amrex::Geometry geom;

    JlGeometry() = default;
    explicit JlGeometry(const amrex::Geometry& g) : geom(g) {}
    JlGeometry(const JlBox& dom, const JlRealBox& rb, int coord,
               bool px, bool py, bool pz) {
        amrex::Array<int, AMREX_SPACEDIM> is_per{int(px), int(py), int(pz)};
        geom = amrex::Geometry(dom.box, rb.rbox, coord, is_per);
    }

    double cell_size(int dir) const { return geom.CellSize(dir); }
    double inv_cell_size(int dir) const { return geom.InvCellSize(dir); }
    double prob_lo(int dir) const { return geom.ProbLo(dir); }
    double prob_hi(int dir) const { return geom.ProbHi(dir); }
    JlBox domain() const { return JlBox(geom.Domain()); }
    JlRealBox prob_domain() const { return JlRealBox(geom.ProbDomain()); }
    bool is_periodic(int dir) const { return geom.isPeriodic(dir); }
};

struct JlBoxArray {
    amrex::BoxArray ba;

    JlBoxArray() = default;
    explicit JlBoxArray(const amrex::BoxArray& b) : ba(b) {}
    explicit JlBoxArray(const JlBox& bx) : ba(bx.box) {}

    ~JlBoxArray() {
        if (!amrex::Initialized()) {
            new (&ba) amrex::BoxArray();
        }
    }

    void max_size(int mx, int my, int mz) {
        ba.maxSize(amrex::IntVect(mx, my, mz));
    }
    JlBoxArray convert_to_nodal(int nx, int ny, int nz) const {
        return JlBoxArray(amrex::convert(ba, amrex::IntVect(nx, ny, nz)));
    }
    int64_t size() const { return ba.size(); }
    bool empty() const { return ba.empty(); }
    JlBox get_box(int64_t i) const { return JlBox(ba[i]); }
    bool contains_cell(int i, int j, int k) const {
        return ba.contains(amrex::IntVect(i, j, k));
    }
};

struct JlDistributionMapping {
    amrex::DistributionMapping dm;

    JlDistributionMapping() = default;
    explicit JlDistributionMapping(const amrex::DistributionMapping& d) : dm(d) {}
    explicit JlDistributionMapping(const JlBoxArray& ba) : dm(ba.ba) {}

    ~JlDistributionMapping() {
        if (!amrex::Initialized()) {
            new (&dm) amrex::DistributionMapping();
        }
    }

    int64_t size() const { return dm.size(); }
    int get_rank(int64_t i) const { return dm[i]; }
};

struct JlMultiFab {
    std::unique_ptr<amrex::MultiFab> mf;

    JlMultiFab() : mf(std::make_unique<amrex::MultiFab>()) {}
    JlMultiFab(const JlBoxArray& ba, const JlDistributionMapping& dm,
               int ncomp, int nghost)
        : mf(std::make_unique<amrex::MultiFab>(
              ba.ba, dm.dm, ncomp, amrex::IntVect(nghost))) {}
    JlMultiFab(const JlBoxArray& ba, const JlDistributionMapping& dm,
               int ncomp, int ngx, int ngy, int ngz)
        : mf(std::make_unique<amrex::MultiFab>(
              ba.ba, dm.dm, ncomp, amrex::IntVect(ngx, ngy, ngz))) {}

    ~JlMultiFab() {
        if (!amrex::Initialized()) {
            mf.release();
        }
    }

    int n_comp() const { return mf->nComp(); }
    int n_grow(int dir) const { return mf->nGrow(dir); }
    int64_t size() const { return mf->size(); }
    int64_t local_size() const { return mf->local_size(); }
    JlBoxArray box_array() const { return JlBoxArray(mf->boxArray()); }
    JlDistributionMapping dmap() const { return JlDistributionMapping(mf->DistributionMap()); }

    void set_val(double val) { mf->setVal(val); }
    void set_val_comp(double val, int icomp, int ncomp, int nghost) {
        mf->setVal(val, icomp, ncomp, nghost);
    }
    void plus_val(double val, int icomp, int ncomp, int nghost) {
        mf->plus(val, icomp, ncomp, nghost);
    }
    void mult_val(double val, int icomp, int ncomp, int nghost) {
        mf->mult(val, icomp, ncomp, nghost);
    }
    void add_from(const JlMultiFab& src, int scomp, int dcomp, int ncomp, int nghost) {
        amrex::MultiFab::Add(*mf, *src.mf, scomp, dcomp, ncomp, nghost);
    }
    void copy_from(const JlMultiFab& src, int scomp, int dcomp, int ncomp,
                   int nghost) {
        amrex::MultiFab::Copy(*mf, *src.mf, scomp, dcomp, ncomp,
                              amrex::IntVect(nghost));
    }
    void parallel_copy_from(const JlMultiFab& src, int scomp, int dcomp, int ncomp,
                            int snghost, int dnghost, const JlGeometry& geom) {
        mf->ParallelCopy(*src.mf, scomp, dcomp, ncomp,
                         amrex::IntVect(snghost), amrex::IntVect(dnghost),
                         geom.geom.periodicity());
    }
    void fill_boundary(const JlGeometry& geom) {
        mf->FillBoundary(geom.geom.periodicity());
    }
    void fill_boundary_comp(int icomp, int ncomp, const JlGeometry& geom) {
        mf->FillBoundary(icomp, ncomp, geom.geom.periodicity());
    }
    void sum_boundary(const JlGeometry& geom) {
        mf->SumBoundary(geom.geom.periodicity());
    }
    void sum_boundary_comp(int icomp, int ncomp, int snghost, int dnghost,
                           const JlGeometry& geom) {
        mf->SumBoundary(icomp, ncomp, amrex::IntVect(snghost),
                        amrex::IntVect(dnghost), geom.geom.periodicity());
    }
    double min_val(int comp, int nghost) const { return mf->min(comp, nghost); }
    double max_val(int comp, int nghost) const { return mf->max(comp, nghost); }
    double norm0(int comp, int nghost) const { return mf->norm0(comp, nghost); }
    double norm1(int comp, const JlGeometry& geom) const {
        return mf->norm1(comp, geom.geom.periodicity());
    }
    double norm2(int comp, const JlGeometry& geom) const {
        return mf->norm2(comp, geom.geom.periodicity());
    }
    double sum_val(int comp, bool local) const { return mf->sum(comp, local); }
};

struct JliMultiFab {
    std::unique_ptr<amrex::iMultiFab> imf;

    JliMultiFab() : imf(std::make_unique<amrex::iMultiFab>()) {}
    JliMultiFab(const JlBoxArray& ba, const JlDistributionMapping& dm,
                int ncomp, int nghost)
        : imf(std::make_unique<amrex::iMultiFab>(
              ba.ba, dm.dm, ncomp, amrex::IntVect(nghost))) {}
    JliMultiFab(const JlBoxArray& ba, const JlDistributionMapping& dm,
                int ncomp, int ngx, int ngy, int ngz)
        : imf(std::make_unique<amrex::iMultiFab>(
              ba.ba, dm.dm, ncomp, amrex::IntVect(ngx, ngy, ngz))) {}

    ~JliMultiFab() {
        if (!amrex::Initialized()) {
            imf.release();
        }
    }

    int n_comp() const { return imf->nComp(); }
    int n_grow(int dir) const { return imf->nGrow(dir); }
    int64_t size() const { return imf->size(); }
    int64_t local_size() const { return imf->local_size(); }
    JlBoxArray box_array() const { return JlBoxArray(imf->boxArray()); }
    JlDistributionMapping dmap() const { return JlDistributionMapping(imf->DistributionMap()); }

    void set_val(int val) { imf->setVal(val); }
    void copy_from(const JliMultiFab& src, int scomp, int dcomp, int ncomp,
                   int nghost) {
        amrex::iMultiFab::Copy(*imf, *src.imf, scomp, dcomp, ncomp,
                               amrex::IntVect(nghost));
    }
    void fill_boundary(const JlGeometry& geom) {
        imf->FillBoundary(geom.geom.periodicity());
    }
    void sum_boundary(const JlGeometry& geom) {
        imf->SumBoundary(geom.geom.periodicity());
    }
    int min_val(int comp, int nghost) const { return imf->min(comp, nghost); }
    int max_val(int comp, int nghost) const { return imf->max(comp, nghost); }
};

struct JlTagBoxArray {
    std::unique_ptr<amrex::TagBoxArray> tags;

    JlTagBoxArray() : tags(nullptr) {}
    JlTagBoxArray(const JlBoxArray& ba, const JlDistributionMapping& dm, int nghost = 0)
        : tags(std::make_unique<amrex::TagBoxArray>(ba.ba, dm.dm, nghost)) {}

    ~JlTagBoxArray() {
        if (!amrex::Initialized()) {
            tags.release();
        }
    }

    int64_t size() const { return tags ? tags->size() : 0; }
    int64_t local_size() const { return tags ? tags->local_size() : 0; }
    JlBoxArray box_array() const { return JlBoxArray(tags->boxArray()); }
    JlDistributionMapping dmap() const { return JlDistributionMapping(tags->DistributionMap()); }
    void set_val(int val) {
        if (tags) {
            tags->setVal(static_cast<char>(val));
        }
    }
};

struct JlMFIter {
    std::unique_ptr<amrex::MFIter> mfi;

    JlMFIter(const JlMultiFab& mf, bool do_tiling)
        : mfi(std::make_unique<amrex::MFIter>(*mf.mf, do_tiling)) {}
    JlMFIter(const JliMultiFab& imf, bool do_tiling)
        : mfi(std::make_unique<amrex::MFIter>(*imf.imf, do_tiling)) {}
    JlMFIter(const JlTagBoxArray& tb, bool do_tiling)
        : mfi(std::make_unique<amrex::MFIter>(*tb.tags, do_tiling)) {}

    ~JlMFIter() {
        if (!amrex::Initialized()) {
            mfi.release();
        }
    }

    bool is_valid() const { return mfi && mfi->isValid(); }
    void next() { ++(*mfi); }
    void close() { mfi.reset(); }
    int index() const { return mfi->index(); }
    int local_tile_index() const { return mfi->LocalTileIndex(); }
    JlBox valid_box() const { return JlBox(mfi->validbox()); }
    JlBox tile_box() const { return JlBox(mfi->tilebox()); }
    JlBox fab_box() const { return JlBox(mfi->fabbox()); }

    int fab_lo(const JlMultiFab& mf, int dir) const {
        return (*mf.mf)[*mfi].box().smallEnd(dir);
    }
    int fab_hi(const JlMultiFab& mf, int dir) const {
        return (*mf.mf)[*mfi].box().bigEnd(dir);
    }
    int ifab_lo(const JliMultiFab& imf, int dir) const {
        return (*imf.imf)[*mfi].box().smallEnd(dir);
    }
    int ifab_hi(const JliMultiFab& imf, int dir) const {
        return (*imf.imf)[*mfi].box().bigEnd(dir);
    }
    int tfab_lo(const JlTagBoxArray& tb, int dir) const {
        return (*tb.tags)[*mfi].box().smallEnd(dir);
    }
    int tfab_hi(const JlTagBoxArray& tb, int dir) const {
        return (*tb.tags)[*mfi].box().bigEnd(dir);
    }

    double* data_ptr(JlMultiFab& mf) const {
        return (*mf.mf)[*mfi].dataPtr();
    }
    int* idata_ptr(JliMultiFab& imf) const {
        return (*imf.imf)[*mfi].dataPtr();
    }
    int8_t* tdata_ptr(JlTagBoxArray& tb) const {
        return reinterpret_cast<int8_t*>((*tb.tags)[*mfi].dataPtr());
    }
};

// ============================================================================
// BCRec
// ============================================================================

struct JlBCRec {
    amrex::BCRec bcr;

    JlBCRec()
        : bcr(amrex::BCType::int_dir, amrex::BCType::int_dir, amrex::BCType::int_dir,
              amrex::BCType::int_dir, amrex::BCType::int_dir, amrex::BCType::int_dir) {}
    explicit JlBCRec(const amrex::BCRec& b) : bcr(b) {}
    JlBCRec(int lox, int loy, int loz, int hix, int hiy, int hiz)
        : bcr(lox, loy, loz, hix, hiy, hiz) {}

    int lo(int dir) const { return bcr.lo(dir); }
    int hi(int dir) const { return bcr.hi(dir); }
    void set_lo(int dir, int bc_val) { bcr.setLo(dir, bc_val); }
    void set_hi(int dir, int bc_val) { bcr.setHi(dir, bc_val); }
};

// ============================================================================
// AmrCore subclass for multi-level AMR hierarchy management
// ============================================================================

using ErrorEstCallbackFn = void (*)(int, uint64_t, double);

class AmrCoreImpl : public amrex::AmrCore {
private:
    static amrex::Geometry make_base_geom(const JlRealBox& rb, int nx, int ny, int nz,
                                          int coord, bool px, bool py, bool pz) {
        amrex::Box dom(amrex::IntVect(0, 0, 0), amrex::IntVect(nx - 1, ny - 1, nz - 1));
        amrex::Array<int, AMREX_SPACEDIM> is_per{int(px), int(py), int(pz)};
        return amrex::Geometry(dom, rb.rbox, coord, is_per);
    }

    static amrex::AmrInfo make_amr_info(int max_level_in, int nz,
                                        int max_grid_x, int max_grid_y, int max_grid_z,
                                        int block_x, int block_y, int block_z,
                                        double grid_eff) {
        amrex::AmrInfo info;
        info.max_level = max_level_in;
        info.ref_ratio.assign(std::max(1, max_level_in),
                              amrex::IntVect(2, 2, (nz == 1) ? 1 : 2));
        info.blocking_factor.assign(max_level_in + 1,
                                    amrex::IntVect(block_x, block_y, (nz == 1) ? 1 : block_z));
        info.max_grid_size.assign(max_level_in + 1,
                                  amrex::IntVect(max_grid_x, max_grid_y, max_grid_z));
        info.grid_eff = grid_eff;
        return info;
    }

public:
    std::vector<std::vector<amrex::RealBox>> refine_boxes;
    std::vector<std::unique_ptr<amrex::iMultiFab>> tag_masks;
    std::vector<std::unique_ptr<amrex::TagBoxArray>> tag_boxes;

    AmrCoreImpl(const JlRealBox& rb, int max_level_in,
                int nx, int ny, int nz, int coord,
                bool px, bool py, bool pz,
                int max_grid_x, int max_grid_y, int max_grid_z,
                int block_x, int block_y, int block_z,
                double grid_eff)
        : amrex::AmrCore(
              make_base_geom(rb, nx, ny, nz, coord, px, py, pz),
              make_amr_info(max_level_in, nz, max_grid_x, max_grid_y, max_grid_z,
                            block_x, block_y, block_z, grid_eff)) {
        refine_boxes.resize(std::max(1, max_level_in));
        tag_masks.resize(std::max(1, max_level_in));
        tag_boxes.resize(std::max(1, max_level_in));
    }

    ~AmrCoreImpl() override {
        for (auto& m : tag_masks) {
            if (!amrex::Initialized()) {
                m.release();
            } else {
                m.reset();
            }
        }
        for (auto& t : tag_boxes) {
            if (!amrex::Initialized()) {
                t.release();
            } else {
                t.reset();
            }
        }
    }

    void set_box_array_pub(int lev, const amrex::BoxArray& ba) { SetBoxArray(lev, ba); }
    void set_dmap_pub(int lev, const amrex::DistributionMapping& dm) { SetDistributionMap(lev, dm); }
    void set_finest_level_pub(int lev) { SetFinestLevel(lev); }

protected:
    void MakeNewLevelFromScratch(int lev, amrex::Real /*time*/,
                                 const amrex::BoxArray& ba,
                                 const amrex::DistributionMapping& dm) override {
        SetBoxArray(lev, ba);
        SetDistributionMap(lev, dm);
    }

    void MakeNewLevelFromCoarse(int lev, amrex::Real /*time*/,
                                const amrex::BoxArray& ba,
                                const amrex::DistributionMapping& dm) override {
        SetBoxArray(lev, ba);
        SetDistributionMap(lev, dm);
    }

    void RemakeLevel(int lev, amrex::Real /*time*/,
                     const amrex::BoxArray& ba,
                     const amrex::DistributionMapping& dm) override {
        SetBoxArray(lev, ba);
        SetDistributionMap(lev, dm);
    }

    void ClearLevel(int lev) override {
        ClearBoxArray(lev);
        ClearDistributionMap(lev);
    }

    void ErrorEst(int lev, amrex::TagBoxArray& tags, amrex::Real /*time*/,
                  int /*ngrow*/) override {
        if (lev < static_cast<int>(refine_boxes.size()) && !refine_boxes[lev].empty()) {
            const auto& geom_lev = Geom(lev);
            for (amrex::MFIter mfi(tags); mfi.isValid(); ++mfi) {
                const amrex::Box& bx = mfi.validbox();
                auto tag_arr = tags.array(mfi);
                const auto lo = amrex::lbound(bx);
                const auto hi = amrex::ubound(bx);
                for (int k = lo.z; k <= hi.z; ++k) {
                    for (int j = lo.y; j <= hi.y; ++j) {
                        for (int i = lo.x; i <= hi.x; ++i) {
                            amrex::Real xyz[3];
                            geom_lev.CellCenter({i, j, k}, xyz);
                            for (int rlev = lev; rlev < static_cast<int>(refine_boxes.size()); ++rlev) {
                                for (const auto& rbox : refine_boxes[rlev]) {
                                    if (rbox.contains(xyz)) {
                                        tag_arr(i, j, k) = amrex::TagBox::SET;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if (lev >= 0 && lev < static_cast<int>(tag_masks.size()) && tag_masks[lev] != nullptr) {
            const auto& mask = *tag_masks[lev];
            for (amrex::MFIter mfi(tags); mfi.isValid(); ++mfi) {
                const amrex::Box& bx = mfi.validbox() & mask[mfi].box();
                if (bx.ok()) {
                    auto tag_arr = tags.array(mfi);
                    auto m_arr = mask.const_array(mfi);
                    const auto lo = amrex::lbound(bx);
                    const auto hi = amrex::ubound(bx);
                    for (int k = lo.z; k <= hi.z; ++k) {
                        for (int j = lo.y; j <= hi.y; ++j) {
                            for (int i = lo.x; i <= hi.x; ++i) {
                                if (m_arr(i, j, k) > 0) {
                                    tag_arr(i, j, k) = amrex::TagBox::SET;
                                } else if (m_arr(i, j, k) < 0) {
                                    tag_arr(i, j, k) = amrex::TagBox::CLEAR;
                                }
                            }
                        }
                    }
                }
            }
        }
        if (lev >= 0 && lev < static_cast<int>(tag_boxes.size()) && tag_boxes[lev] != nullptr) {
            const auto& tb = *tag_boxes[lev];
            for (amrex::MFIter mfi(tags); mfi.isValid(); ++mfi) {
                const amrex::Box& bx = mfi.validbox() & tb[mfi].box();
                if (bx.ok()) {
                    auto tag_arr = tags.array(mfi);
                    auto src_arr = tb.const_array(mfi);
                    const auto lo = amrex::lbound(bx);
                    const auto hi = amrex::ubound(bx);
                    for (int k = lo.z; k <= hi.z; ++k) {
                        for (int j = lo.y; j <= hi.y; ++j) {
                            for (int i = lo.x; i <= hi.x; ++i) {
                                if (src_arr(i, j, k) == amrex::TagBox::SET) {
                                    tag_arr(i, j, k) = amrex::TagBox::SET;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
};

struct JlAmrCore {
    std::shared_ptr<AmrCoreImpl> core;

    JlAmrCore(const JlRealBox& rb, int max_level_in,
              int nx, int ny, int nz, int coord,
              bool px, bool py, bool pz,
              int max_grid_x, int max_grid_y, int max_grid_z,
              int block_x, int block_y, int block_z,
              double grid_eff)
        : core(std::make_shared<AmrCoreImpl>(
              rb, max_level_in, nx, ny, nz, coord, px, py, pz,
              max_grid_x, max_grid_y, max_grid_z, block_x, block_y, block_z, grid_eff)) {}

    ~JlAmrCore() {
        if (!amrex::Initialized()) {
            new (&core) std::shared_ptr<AmrCoreImpl>();
        }
    }

    void add_refine_box(int lev, double xlo, double ylo, double zlo,
                        double xhi, double yhi, double zhi) {
        if (lev >= 0 && lev < static_cast<int>(core->refine_boxes.size())) {
            core->refine_boxes[lev].emplace_back(
                amrex::RealBox({xlo, ylo, zlo}, {xhi, yhi, zhi}));
        }
    }

    void clear_refine_boxes() {
        for (auto& v : core->refine_boxes) {
            v.clear();
        }
    }

    void set_tag_mask(int lev, const JliMultiFab& mask) {
        if (lev >= 0 && lev < static_cast<int>(core->tag_masks.size())) {
            core->tag_masks[lev] = std::make_unique<amrex::iMultiFab>(
                mask.imf->boxArray(), mask.imf->DistributionMap(), 1, 0);
            amrex::iMultiFab::Copy(*core->tag_masks[lev], *mask.imf, 0, 0, 1, 0);
        }
    }

    void set_tag_boxes(int lev, const JlTagBoxArray& tb) {
        if (lev >= 0 && lev < static_cast<int>(core->tag_boxes.size())) {
            core->tag_boxes[lev] = std::make_unique<amrex::TagBoxArray>(
                tb.tags->boxArray(), tb.tags->DistributionMap(), 0);
            core->tag_boxes[lev]->setVal(amrex::TagBox::CLEAR);
            for (amrex::MFIter mfi(*core->tag_boxes[lev]); mfi.isValid(); ++mfi) {
                const amrex::Box& bx = mfi.validbox() & (*tb.tags)[mfi].box();
                if (bx.ok()) {
                    auto dst = core->tag_boxes[lev]->array(mfi);
                    auto src = tb.tags->const_array(mfi);
                    const auto lo = amrex::lbound(bx);
                    const auto hi = amrex::ubound(bx);
                    for (int k = lo.z; k <= hi.z; ++k) {
                        for (int j = lo.y; j <= hi.y; ++j) {
                            for (int i = lo.x; i <= hi.x; ++i) {
                                dst(i, j, k) = src(i, j, k);
                            }
                        }
                    }
                }
            }
        }
    }

    void clear_tag_masks() {
        for (auto& m : core->tag_masks) {
            m.reset();
        }
        for (auto& t : core->tag_boxes) {
            t.reset();
        }
    }

    void ensure_base_grids() {
        if (core->finestLevel() < 0) {
            amrex::BoxArray ba0 = core->MakeBaseGrids();
            amrex::DistributionMapping dm0 = core->MakeDistributionMap(0, ba0);
            core->set_box_array_pub(0, ba0);
            core->set_dmap_pub(0, dm0);
            core->set_finest_level_pub(0);
        }
    }

    void init_from_scratch(double time) { core->InitFromScratch(time); }
    void regrid_levels(int base_lev, double time) { core->regrid(base_lev, time); }

    int get_finest_level() const { return core->finestLevel(); }
    int get_max_level() const { return core->maxLevel(); }
    JlGeometry get_geom(int lev) const { return JlGeometry(core->Geom(lev)); }
    JlBoxArray get_box_array(int lev) const { return JlBoxArray(core->boxArray(lev)); }
    JlBoxArray get_node_box_array(int lev) const {
        return JlBoxArray(amrex::convert(core->boxArray(lev), amrex::IntVect::TheNodeVector()));
    }
    JlDistributionMapping get_dmap(int lev) const {
        return JlDistributionMapping(core->DistributionMap(lev));
    }
    int get_ref_ratio(int lev, int dir) const {
        return (lev < core->maxLevel()) ? core->refRatio(lev)[dir] : 1;
    }

    void set_box_array(int lev, const JlBoxArray& ba) { core->set_box_array_pub(lev, ba.ba); }
    void set_dmap(int lev, const JlDistributionMapping& dm) {
        core->set_dmap_pub(lev, dm.dm);
    }
    void set_finest_level(int lev) { core->set_finest_level_pub(lev); }
};

// ============================================================================
// Pure-SoA ParticleContainer & ParIter
// ============================================================================

using PureSoAParticleContainer =
    amrex::AmrParticleContainer_impl<amrex::SoAParticle<AMREX_SPACEDIM, 0>,
                                     AMREX_SPACEDIM, 0>;
using PureSoAParIter = amrex::ParIterSoA<AMREX_SPACEDIM, 0>;

struct JlParticleContainer {
    std::shared_ptr<AmrCoreImpl> amr_ref;
    std::unique_ptr<PureSoAParticleContainer> pc;
    int num_runtime_real = 0;
    int num_runtime_int = 0;

    JlParticleContainer(JlAmrCore& amr_core, int n_rt_real, int n_rt_int)
        : amr_ref(amr_core.core),
          pc(std::make_unique<PureSoAParticleContainer>(amr_core.core.get())),
          num_runtime_real(n_rt_real),
          num_runtime_int(n_rt_int) {
        pc->SetSoACompileTimeNames({"x", "y", "z"}, {});
        for (int i = 0; i < n_rt_real; ++i) {
            pc->AddRealComp("r_" + std::to_string(i), true);
        }
        for (int i = 0; i < n_rt_int; ++i) {
            pc->AddIntComp("i_" + std::to_string(i), true);
        }
        pc->reserveData();
        pc->resizeData();
    }

    JlParticleContainer(const JlGeometry& geom, const JlDistributionMapping& dm,
                        const JlBoxArray& ba, int n_rt_real, int n_rt_int)
        : amr_ref(nullptr),
          pc(std::make_unique<PureSoAParticleContainer>(
              amrex::Vector<amrex::Geometry>{geom.geom},
              amrex::Vector<amrex::DistributionMapping>{dm.dm},
              amrex::Vector<amrex::BoxArray>{ba.ba},
              amrex::Vector<int>{})),
          num_runtime_real(n_rt_real),
          num_runtime_int(n_rt_int) {
        pc->SetSoACompileTimeNames({"x", "y", "z"}, {});
        for (int i = 0; i < n_rt_real; ++i) {
            pc->AddRealComp("r_" + std::to_string(i), true);
        }
        for (int i = 0; i < n_rt_int; ++i) {
            pc->AddIntComp("i_" + std::to_string(i), true);
        }
        pc->reserveData();
        pc->resizeData();
    }

    ~JlParticleContainer() {
        if (!amrex::Initialized()) {
            pc.release();
            new (&amr_ref) std::shared_ptr<AmrCoreImpl>();
        } else {
            pc.reset();
            amr_ref.reset();
        }
    }

    void reserve_and_resize_data() {
        pc->reserveData();
        pc->resizeData();
    }

    // Ensure a tile exists at (lev, grid_id, tile_id) and return its current particle count
    int64_t ensure_tile(int lev, int grid_id, int tile_id) {
        auto& ptile = pc->DefineAndReturnParticleTile(lev, grid_id, tile_id);
        return ptile.numParticles();
    }

    void resize_tile(int lev, int grid_id, int tile_id, int64_t new_size) {
        auto& ptile = pc->DefineAndReturnParticleTile(lev, grid_id, tile_id);
        ptile.resize(new_size);
    }

    int64_t tile_num_particles(int lev, int grid_id, int tile_id) {
        auto& ptile = pc->DefineAndReturnParticleTile(lev, grid_id, tile_id);
        return ptile.numParticles();
    }

    uint64_t* tile_idcpu_ptr(int lev, int grid_id, int tile_id) {
        auto& ptile = pc->DefineAndReturnParticleTile(lev, grid_id, tile_id);
        return ptile.GetStructOfArrays().GetIdCPUData().dataPtr();
    }

    double* tile_pos_ptr(int lev, int grid_id, int tile_id, int dir) {
        auto& ptile = pc->DefineAndReturnParticleTile(lev, grid_id, tile_id);
        return ptile.GetStructOfArrays().GetRealData(dir).dataPtr();
    }

    double* tile_real_ptr(int lev, int grid_id, int tile_id, int comp) {
        auto& ptile = pc->DefineAndReturnParticleTile(lev, grid_id, tile_id);
        return ptile.GetStructOfArrays().GetRealData(AMREX_SPACEDIM + comp).dataPtr();
    }

    int* tile_int_ptr(int lev, int grid_id, int tile_id, int comp) {
        auto& ptile = pc->DefineAndReturnParticleTile(lev, grid_id, tile_id);
        return ptile.GetStructOfArrays().GetIntData(comp).dataPtr();
    }

    void redistribute(int lev_min, int lev_max, int nghost) {
        pc->Redistribute(lev_min, lev_max, nghost);
    }

    int64_t total_number_of_particles(bool only_valid, bool local) const {
        return pc->TotalNumberOfParticles(only_valid, local);
    }

    int64_t number_of_particles_at_level(int lev, bool only_valid, bool local) const {
        return pc->NumberOfParticlesAtLevel(lev, only_valid, local);
    }

    void clear_particles() {
        pc->clearParticles();
    }

    void write_plotfile(const std::string& dir, const std::string& name) const {
        if (amrex::ParallelDescriptor::IOProcessor() && !dir.empty()) {
            amrex::UtilCreateDirectory(dir, 0755);
        }
        amrex::ParallelDescriptor::Barrier();
        pc->WritePlotFile(dir, name);
    }

    void checkpoint(const std::string& dir, const std::string& name) const {
        if (amrex::ParallelDescriptor::IOProcessor() && !dir.empty()) {
            amrex::UtilCreateDirectory(dir, 0755);
        }
        amrex::ParallelDescriptor::Barrier();
        pc->Checkpoint(dir, name);
    }

    void restart(const std::string& dir, const std::string& file) {
        pc->Restart(dir, file);
    }

    void write_plotfile_hdf5(const std::string& dir, const std::string& name,
                             const std::string& compression) const {
#ifdef AMREX_USE_HDF5
        if (amrex::ParallelDescriptor::IOProcessor() && !dir.empty()) {
            amrex::UtilCreateDirectory(dir, 0755);
        }
        amrex::ParallelDescriptor::Barrier();
        pc->WritePlotFileHDF5(dir, name, compression);
#else
        amrex::ignore_unused(dir, name, compression);
        amrex::Abort("AMReX was not built with HDF5 support (AMREX_USE_HDF5)");
#endif
    }

    void checkpoint_hdf5(const std::string& dir, const std::string& name,
                         const std::string& compression) const {
#ifdef AMREX_USE_HDF5
        if (amrex::ParallelDescriptor::IOProcessor() && !dir.empty()) {
            amrex::UtilCreateDirectory(dir, 0755);
        }
        amrex::ParallelDescriptor::Barrier();
        pc->CheckpointHDF5(dir, name, true, amrex::Vector<std::string>{},
                           amrex::Vector<std::string>{}, compression);
#else
        amrex::ignore_unused(dir, name, compression);
        amrex::Abort("AMReX was not built with HDF5 support (AMREX_USE_HDF5)");
#endif
    }

    void restart_hdf5(const std::string& dir, const std::string& file) {
#ifdef AMREX_USE_HDF5
        std::string base = dir;
        if (!base.empty() && base.back() != '/') {
            base += '/';
        }
        std::string direct_h5 = base + file + ".h5";
        std::string subdir_h5 = base + file + "/" + file + ".h5";
        if (!amrex::FileExists(direct_h5) && amrex::FileExists(subdir_h5)) {
            pc->RestartHDF5(base + file, file);
        } else {
            pc->RestartHDF5(dir, file);
        }
#else
        amrex::ignore_unused(dir, file);
        amrex::Abort("AMReX was not built with HDF5 support (AMREX_USE_HDF5)");
#endif
    }
};

struct JlParIter {
    std::unique_ptr<PureSoAParIter> pti;

    JlParIter(JlParticleContainer& pc, int lev)
        : pti(std::make_unique<PureSoAParIter>(*pc.pc, lev)) {}

    ~JlParIter() {
        if (!amrex::Initialized()) {
            pti.release();
        }
    }

    bool is_valid() const { return pti && pti->isValid(); }
    void next() { ++(*pti); }
    void close() { pti.reset(); }
    int index() const { return pti->index(); }
    int local_tile_index() const { return pti->LocalTileIndex(); }
    JlBox valid_box() const { return JlBox(pti->validbox()); }
    JlBox tile_box() const { return JlBox(pti->tilebox()); }
    int64_t num_particles() const { return pti->numParticles(); }

    void resize_tile(int64_t new_size) {
        pti->GetParticleTile().resize(new_size);
    }

    uint64_t* idcpu_ptr() {
        return pti->GetStructOfArrays().GetIdCPUData().dataPtr();
    }

    double* pos_ptr(int dir) {
        return pti->GetStructOfArrays().GetRealData(dir).dataPtr();
    }

    double* real_ptr(int comp) {
        return pti->GetStructOfArrays().GetRealData(AMREX_SPACEDIM + comp).dataPtr();
    }

    int* int_ptr(int comp) {
        return pti->GetStructOfArrays().GetIntData(comp).dataPtr();
    }
};

// ============================================================================
// Flux Registers (YAFluxRegister & EdgeFluxRegister)
// ============================================================================

struct JlYAFluxRegister {
    std::unique_ptr<amrex::YAFluxRegister> fr;

    JlYAFluxRegister() : fr(std::make_unique<amrex::YAFluxRegister>()) {}
    JlYAFluxRegister(const JlBoxArray& fba, const JlBoxArray& cba,
                     const JlDistributionMapping& fdm, const JlDistributionMapping& cdm,
                     const JlGeometry& fgeom, const JlGeometry& cgeom,
                     int rx, int ry, int rz, int fine_lev, int nvar)
        : fr(std::make_unique<amrex::YAFluxRegister>(
              fba.ba, cba.ba, fdm.dm, cdm.dm, fgeom.geom, cgeom.geom,
              amrex::IntVect(rx, ry, rz), fine_lev, nvar)) {}

    ~JlYAFluxRegister() {
        if (!amrex::Initialized()) {
            fr.release();
        }
    }

    void reset() { fr->reset(); }

    void crse_add(const JlMFIter& mfi,
                  const JlMultiFab& fx, const JlMultiFab& fy, const JlMultiFab& fz,
                  double dx, double dy, double dz, double dt,
                  int srccomp, int destcomp, int numcomp) {
        std::array<amrex::FArrayBox const*, AMREX_SPACEDIM> flux{
            &((*fx.mf)[*mfi.mfi]),
            &((*fy.mf)[*mfi.mfi]),
            &((*fz.mf)[*mfi.mfi])
        };
        amrex::Real dx_arr[AMREX_SPACEDIM] = {dx, dy, dz};
        fr->CrseAdd(*mfi.mfi, flux, dx_arr, dt, srccomp, destcomp, numcomp,
                    amrex::RunOn::Host);
    }

    void fine_add(const JlMFIter& mfi,
                  const JlMultiFab& fx, const JlMultiFab& fy, const JlMultiFab& fz,
                  double dx, double dy, double dz, double dt,
                  int srccomp, int destcomp, int numcomp) {
        std::array<amrex::FArrayBox const*, AMREX_SPACEDIM> flux{
            &((*fx.mf)[*mfi.mfi]),
            &((*fy.mf)[*mfi.mfi]),
            &((*fz.mf)[*mfi.mfi])
        };
        amrex::Real dx_arr[AMREX_SPACEDIM] = {dx, dy, dz};
        fr->FineAdd(*mfi.mfi, flux, dx_arr, dt, srccomp, destcomp, numcomp,
                    amrex::RunOn::Host);
    }

    void reflux(JlMultiFab& state, int srccomp, int destcomp, int numcomp) {
        fr->Reflux(*state.mf, srccomp, destcomp, numcomp);
    }

    bool crse_has_work(const JlMFIter& mfi) const {
        return fr->CrseHasWork(*mfi.mfi);
    }

    bool fine_has_work(const JlMFIter& mfi) const {
        return fr->FineHasWork(*mfi.mfi);
    }
};

struct JlEdgeFluxRegister {
    std::unique_ptr<amrex::EdgeFluxRegister> efr;

    JlEdgeFluxRegister() : efr(std::make_unique<amrex::EdgeFluxRegister>()) {}
    JlEdgeFluxRegister(const JlBoxArray& fba, const JlBoxArray& cba,
                       const JlDistributionMapping& fdm, const JlDistributionMapping& cdm,
                       const JlGeometry& fgeom, const JlGeometry& cgeom,
                       int nvar)
        : efr(std::make_unique<amrex::EdgeFluxRegister>(
              fba.ba, cba.ba, fdm.dm, cdm.dm, fgeom.geom, cgeom.geom, nvar)) {}

    ~JlEdgeFluxRegister() {
        if (!amrex::Initialized()) {
            efr.release();
        }
    }

    void reset() { efr->reset(); }

    void crse_add(const JlMFIter& mfi,
                  const JlMultiFab& Ex, const JlMultiFab& Ey, const JlMultiFab& Ez,
                  double dt_crse) {
        amrex::Array<amrex::FArrayBox const*, AMREX_SPACEDIM> E_crse{
            &((*Ex.mf)[*mfi.mfi]),
            &((*Ey.mf)[*mfi.mfi]),
            &((*Ez.mf)[*mfi.mfi])
        };
        efr->CrseAdd(*mfi.mfi, E_crse, dt_crse);
    }

    void fine_add(const JlMFIter& mfi,
                  const JlMultiFab& Ex, const JlMultiFab& Ey, const JlMultiFab& Ez,
                  double dt_fine) {
        amrex::Array<amrex::FArrayBox const*, AMREX_SPACEDIM> E_fine{
            &((*Ex.mf)[*mfi.mfi]),
            &((*Ey.mf)[*mfi.mfi]),
            &((*Ez.mf)[*mfi.mfi])
        };
        efr->FineAdd(*mfi.mfi, E_fine, dt_fine);
    }

    void reflux(JlMultiFab& Bx, JlMultiFab& By, JlMultiFab& Bz) const {
        amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM> B_crse{
            Bx.mf.get(), By.mf.get(), Bz.mf.get()
        };
        efr->Reflux(B_crse);
    }
};

// ============================================================================
// Linear Solvers (LPInfo, MLPoisson, MLABecLaplacian, MLNodeLaplacian, MLMG)
// ============================================================================

struct JlLPInfo {
    amrex::LPInfo info;

    JlLPInfo() = default;
    JlLPInfo(int max_coarsening_level, bool do_agg, bool do_con,
             int agg_grid_size, int con_grid_size,
             bool has_metric_term, int hidden_direction) {
        info.setMaxCoarseningLevel(max_coarsening_level)
            .setAgglomeration(do_agg)
            .setConsolidation(do_con)
            .setAgglomerationGridSize(agg_grid_size)
            .setConsolidationGridSize(con_grid_size)
            .setMetricTerm(has_metric_term)
            .setHiddenDirection(hidden_direction);
    }

    void set_max_coarsening_level(int n) { info.setMaxCoarseningLevel(n); }
    void set_agglomeration(bool x) { info.setAgglomeration(x); }
    void set_consolidation(bool x) { info.setConsolidation(x); }
    void set_agg_grid_size(int x) { info.setAgglomerationGridSize(x); }
    void set_con_grid_size(int x) { info.setConsolidationGridSize(x); }
    void set_metric_term(bool x) { info.setMetricTerm(x); }
    void set_hidden_direction(int n) { info.setHiddenDirection(n); }
};

struct JlMLPoisson {
    std::unique_ptr<amrex::MLPoisson> op;
    amrex::Vector<amrex::Geometry> staged_geom;
    amrex::Vector<amrex::BoxArray> staged_ba;
    amrex::Vector<amrex::DistributionMapping> staged_dm;

    JlMLPoisson() : op(std::make_unique<amrex::MLPoisson>()) {}
    JlMLPoisson(const JlGeometry& geom, const JlBoxArray& ba,
                const JlDistributionMapping& dm, const JlLPInfo& info)
        : op(std::make_unique<amrex::MLPoisson>(
              amrex::Vector<amrex::Geometry>{geom.geom},
              amrex::Vector<amrex::BoxArray>{ba.ba},
              amrex::Vector<amrex::DistributionMapping>{dm.dm},
              info.info)) {}

    ~JlMLPoisson() {
        if (!amrex::Initialized()) {
            op.release();
            new (&staged_geom) amrex::Vector<amrex::Geometry>();
            new (&staged_ba) amrex::Vector<amrex::BoxArray>();
            new (&staged_dm) amrex::Vector<amrex::DistributionMapping>();
        }
    }

    void stage_clear() {
        staged_geom.clear();
        staged_ba.clear();
        staged_dm.clear();
    }
    void stage_push_level(const JlGeometry& g, const JlBoxArray& ba,
                          const JlDistributionMapping& dm) {
        staged_geom.push_back(g.geom);
        staged_ba.push_back(ba.ba);
        staged_dm.push_back(dm.dm);
    }
    void define_staged(const JlLPInfo& info) {
        op = std::make_unique<amrex::MLPoisson>(staged_geom, staged_ba, staged_dm, info.info);
        stage_clear();
    }

    void set_domain_bc(int lox, int loy, int loz, int hix, int hiy, int hiz) {
        op->setDomainBC(
            {static_cast<amrex::LinOpBCType>(lox),
             static_cast<amrex::LinOpBCType>(loy),
             static_cast<amrex::LinOpBCType>(loz)},
            {static_cast<amrex::LinOpBCType>(hix),
             static_cast<amrex::LinOpBCType>(hiy),
             static_cast<amrex::LinOpBCType>(hiz)});
    }
    void set_domain_bc_loc(double lox, double loy, double loz,
                           double hix, double hiy, double hiz) {
        op->setDomainBCLoc({lox, loy, loz}, {hix, hiy, hiz});
    }
    void set_level_bc_homog(int amrlev) { op->setLevelBC(amrlev, nullptr); }
    void set_level_bc(int amrlev, const JlMultiFab& bcdata) {
        op->setLevelBC(amrlev, bcdata.mf.get());
    }
    void set_level_bc_robin(int amrlev, const JlMultiFab& bcdata,
                            const JlMultiFab& ra, const JlMultiFab& rb, const JlMultiFab& rf) {
        op->setLevelBC(amrlev, bcdata.mf.get(), ra.mf.get(), rb.mf.get(), rf.mf.get());
    }
    void set_coarse_fine_bc(const JlMultiFab& crse, int rx, int ry, int rz, int bc_type) {
        op->setCoarseFineBC(crse.mf.get(), amrex::IntVect(rx, ry, rz),
                            static_cast<amrex::LinOpBCType>(bc_type));
    }
    void set_coarse_fine_bc_homog(int rx, int ry, int rz, int bc_type) {
        op->setCoarseFineBC(static_cast<const amrex::MultiFab*>(nullptr),
                            amrex::IntVect(rx, ry, rz),
                            static_cast<amrex::LinOpBCType>(bc_type));
    }
    void set_max_order(int o) { op->setMaxOrder(o); }
    int get_max_order() const { return op->getMaxOrder(); }
    void set_verbose(int v) { op->setVerbose(v); }
    void set_enforce_singular_solvable(bool o) { op->setEnforceSingularSolvable(o); }
};

struct JlMLABecLaplacian {
    std::unique_ptr<amrex::MLABecLaplacian> op;
    amrex::Vector<amrex::Geometry> staged_geom;
    amrex::Vector<amrex::BoxArray> staged_ba;
    amrex::Vector<amrex::DistributionMapping> staged_dm;

    JlMLABecLaplacian() : op(std::make_unique<amrex::MLABecLaplacian>()) {}
    JlMLABecLaplacian(const JlGeometry& geom, const JlBoxArray& ba,
                      const JlDistributionMapping& dm, const JlLPInfo& info, int ncomp)
        : op(std::make_unique<amrex::MLABecLaplacian>(
              amrex::Vector<amrex::Geometry>{geom.geom},
              amrex::Vector<amrex::BoxArray>{ba.ba},
              amrex::Vector<amrex::DistributionMapping>{dm.dm},
              info.info, amrex::Vector<amrex::FabFactory<amrex::FArrayBox> const*>{}, ncomp)) {}

    ~JlMLABecLaplacian() {
        if (!amrex::Initialized()) {
            op.release();
            new (&staged_geom) amrex::Vector<amrex::Geometry>();
            new (&staged_ba) amrex::Vector<amrex::BoxArray>();
            new (&staged_dm) amrex::Vector<amrex::DistributionMapping>();
        }
    }

    void stage_clear() {
        staged_geom.clear();
        staged_ba.clear();
        staged_dm.clear();
    }
    void stage_push_level(const JlGeometry& g, const JlBoxArray& ba,
                          const JlDistributionMapping& dm) {
        staged_geom.push_back(g.geom);
        staged_ba.push_back(ba.ba);
        staged_dm.push_back(dm.dm);
    }
    void define_staged(const JlLPInfo& info, int ncomp) {
        op = std::make_unique<amrex::MLABecLaplacian>(
            staged_geom, staged_ba, staged_dm, info.info,
            amrex::Vector<amrex::FabFactory<amrex::FArrayBox> const*>{}, ncomp);
        stage_clear();
    }

    void set_domain_bc(int lox, int loy, int loz, int hix, int hiy, int hiz) {
        op->setDomainBC(
            {static_cast<amrex::LinOpBCType>(lox),
             static_cast<amrex::LinOpBCType>(loy),
             static_cast<amrex::LinOpBCType>(loz)},
            {static_cast<amrex::LinOpBCType>(hix),
             static_cast<amrex::LinOpBCType>(hiy),
             static_cast<amrex::LinOpBCType>(hiz)});
    }
    void set_domain_bc_loc(double lox, double loy, double loz,
                           double hix, double hiy, double hiz) {
        op->setDomainBCLoc({lox, loy, loz}, {hix, hiy, hiz});
    }
    void set_level_bc_homog(int amrlev) { op->setLevelBC(amrlev, nullptr); }
    void set_level_bc(int amrlev, const JlMultiFab& bcdata) {
        op->setLevelBC(amrlev, bcdata.mf.get());
    }
    void set_level_bc_robin(int amrlev, const JlMultiFab& bcdata,
                            const JlMultiFab& ra, const JlMultiFab& rb, const JlMultiFab& rf) {
        op->setLevelBC(amrlev, bcdata.mf.get(), ra.mf.get(), rb.mf.get(), rf.mf.get());
    }
    void set_coarse_fine_bc(const JlMultiFab& crse, int rx, int ry, int rz, int bc_type) {
        op->setCoarseFineBC(crse.mf.get(), amrex::IntVect(rx, ry, rz),
                            static_cast<amrex::LinOpBCType>(bc_type));
    }
    void set_coarse_fine_bc_homog(int rx, int ry, int rz, int bc_type) {
        op->setCoarseFineBC(static_cast<const amrex::MultiFab*>(nullptr),
                            amrex::IntVect(rx, ry, rz),
                            static_cast<amrex::LinOpBCType>(bc_type));
    }
    void set_max_order(int o) { op->setMaxOrder(o); }
    int get_max_order() const { return op->getMaxOrder(); }
    void set_verbose(int v) { op->setVerbose(v); }
    void set_enforce_singular_solvable(bool o) { op->setEnforceSingularSolvable(o); }

    void set_scalars(double a, double b) { op->setScalars(a, b); }
    void set_a_coeffs_scalar(int amrlev, double alpha) { op->setACoeffs(amrlev, alpha); }
    void set_a_coeffs(int amrlev, const JlMultiFab& alpha) {
        op->setACoeffs(amrlev, *alpha.mf);
    }
    void set_b_coeffs_scalar(int amrlev, double beta) { op->setBCoeffs(amrlev, beta); }
    void set_b_coeffs(int amrlev, const JlMultiFab& bx, const JlMultiFab& by, const JlMultiFab& bz) {
        amrex::Array<amrex::MultiFab const*, AMREX_SPACEDIM> b{
            bx.mf.get(), by.mf.get(), bz.mf.get()
        };
        op->setBCoeffs(amrlev, b);
    }
};

struct JlMLNodeLaplacian {
    std::unique_ptr<amrex::MLNodeLaplacian> op;
    amrex::Vector<amrex::Geometry> staged_geom;
    amrex::Vector<amrex::BoxArray> staged_ba;
    amrex::Vector<amrex::DistributionMapping> staged_dm;

    JlMLNodeLaplacian() : op(std::make_unique<amrex::MLNodeLaplacian>()) {}
    JlMLNodeLaplacian(const JlGeometry& geom, const JlBoxArray& ba,
                      const JlDistributionMapping& dm, const JlLPInfo& info, double const_sigma)
        : op(std::make_unique<amrex::MLNodeLaplacian>(
              amrex::Vector<amrex::Geometry>{geom.geom},
              amrex::Vector<amrex::BoxArray>{ba.ba},
              amrex::Vector<amrex::DistributionMapping>{dm.dm},
              info.info, amrex::Vector<amrex::FabFactory<amrex::FArrayBox> const*>{},
              const_sigma)) {}

    ~JlMLNodeLaplacian() {
        if (!amrex::Initialized()) {
            op.release();
            new (&staged_geom) amrex::Vector<amrex::Geometry>();
            new (&staged_ba) amrex::Vector<amrex::BoxArray>();
            new (&staged_dm) amrex::Vector<amrex::DistributionMapping>();
        }
    }

    void stage_clear() {
        staged_geom.clear();
        staged_ba.clear();
        staged_dm.clear();
    }
    void stage_push_level(const JlGeometry& g, const JlBoxArray& ba,
                          const JlDistributionMapping& dm) {
        staged_geom.push_back(g.geom);
        staged_ba.push_back(ba.ba);
        staged_dm.push_back(dm.dm);
    }
    void define_staged(const JlLPInfo& info, double const_sigma) {
        op = std::make_unique<amrex::MLNodeLaplacian>(
            staged_geom, staged_ba, staged_dm, info.info,
            amrex::Vector<amrex::FabFactory<amrex::FArrayBox> const*>{}, const_sigma);
        stage_clear();
    }

    void set_domain_bc(int lox, int loy, int loz, int hix, int hiy, int hiz) {
        op->setDomainBC(
            {static_cast<amrex::LinOpBCType>(lox),
             static_cast<amrex::LinOpBCType>(loy),
             static_cast<amrex::LinOpBCType>(loz)},
            {static_cast<amrex::LinOpBCType>(hix),
             static_cast<amrex::LinOpBCType>(hiy),
             static_cast<amrex::LinOpBCType>(hiz)});
    }
    void set_max_order(int o) { op->setMaxOrder(o); }
    int get_max_order() const { return op->getMaxOrder(); }
    void set_verbose(int v) { op->setVerbose(v); }
    void set_enforce_singular_solvable(bool o) { op->setEnforceSingularSolvable(o); }

    void set_sigma(int amrlev, const JlMultiFab& sigma) {
        op->setSigma(amrlev, *sigma.mf);
    }
    void set_gauss_seidel(bool flag) { op->setGaussSeidel(flag); }
    void set_harmonic_average(bool flag) { op->setHarmonicAverage(flag); }

    void comp_divergence_1(JlMultiFab& rhs, JlMultiFab& vel) {
        amrex::Vector<amrex::MultiFab*> r{rhs.mf.get()};
        amrex::Vector<amrex::MultiFab*> v{vel.mf.get()};
        op->compDivergence(r, v);
    }
    void update_velocity_1(JlMultiFab& vel, const JlMultiFab& sol) const {
        amrex::Vector<amrex::MultiFab*> v{vel.mf.get()};
        amrex::Vector<amrex::MultiFab const*> s{sol.mf.get()};
        op->updateVelocity(v, s);
    }
};

struct JlMLMG {
    std::unique_ptr<amrex::MLMG> mlmg;
    amrex::Vector<amrex::MultiFab*> staged_sol;
    amrex::Vector<amrex::MultiFab const*> staged_rhs;
    amrex::Vector<amrex::MultiFab*> staged_out;
    amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>> staged_face_vec;

    explicit JlMLMG(JlMLPoisson& p) : mlmg(std::make_unique<amrex::MLMG>(*p.op)) {}
    explicit JlMLMG(JlMLABecLaplacian& abec) : mlmg(std::make_unique<amrex::MLMG>(*abec.op)) {}
    explicit JlMLMG(JlMLNodeLaplacian& nd) : mlmg(std::make_unique<amrex::MLMG>(*nd.op)) {}

    ~JlMLMG() {
        if (!amrex::Initialized()) {
            mlmg.release();
            new (&staged_sol) amrex::Vector<amrex::MultiFab*>();
            new (&staged_rhs) amrex::Vector<amrex::MultiFab const*>();
            new (&staged_out) amrex::Vector<amrex::MultiFab*>();
            new (&staged_face_vec) amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>>();
        }
    }

    void set_verbose(int v) { mlmg->setVerbose(v); }
    void set_bottom_verbose(int v) { mlmg->setBottomVerbose(v); }
    void set_max_iter(int n) { mlmg->setMaxIter(n); }
    void set_max_fmg_iter(int n) { mlmg->setMaxFmgIter(n); }
    void set_fixed_iter(int nit) { mlmg->setFixedIter(nit); }
    void set_pre_smooth(int n) { mlmg->setPreSmooth(n); }
    void set_post_smooth(int n) { mlmg->setPostSmooth(n); }
    void set_final_smooth(int n) { mlmg->setFinalSmooth(n); }
    void set_bottom_smooth(int n) { mlmg->setBottomSmooth(n); }
    void set_bottom_solver(int s) {
        mlmg->setBottomSolver(static_cast<amrex::BottomSolver>(s));
    }
    void set_bottom_max_iter(int n) { mlmg->setBottomMaxIter(n); }
    void set_bottom_tolerance(double t) { mlmg->setBottomTolerance(t); }
    void set_bottom_tolerance_abs(double t) { mlmg->setBottomToleranceAbs(t); }
    void set_final_fill_bc(int flag) { mlmg->setFinalFillBC(flag); }
    int num_amr_levels() const { return mlmg->numAMRLevels(); }
    double get_init_residual() const { return mlmg->getInitResidual(); }
    double get_final_residual() const { return mlmg->getFinalResidual(); }
    double get_init_rhs() const { return mlmg->getInitRHS(); }

    void stage_clear() {
        staged_sol.clear();
        staged_rhs.clear();
        staged_out.clear();
        staged_face_vec.clear();
    }
    void stage_push_sol_rhs(JlMultiFab& sol, const JlMultiFab& rhs) {
        staged_sol.push_back(sol.mf.get());
        staged_rhs.push_back(rhs.mf.get());
    }
    void stage_push_out_sol_rhs(JlMultiFab& out, JlMultiFab& sol, const JlMultiFab& rhs) {
        staged_out.push_back(out.mf.get());
        staged_sol.push_back(sol.mf.get());
        staged_rhs.push_back(rhs.mf.get());
    }
    void stage_push_out_in(JlMultiFab& out, JlMultiFab& in) {
        staged_out.push_back(out.mf.get());
        staged_sol.push_back(in.mf.get());
    }
    void stage_push_face_vec(JlMultiFab& fx, JlMultiFab& fy, JlMultiFab& fz) {
        staged_face_vec.push_back({fx.mf.get(), fy.mf.get(), fz.mf.get()});
    }
    void stage_push_out(JlMultiFab& out) {
        staged_out.push_back(out.mf.get());
    }

    double solve_staged(double tol_rel, double tol_abs) {
        double res = mlmg->solve(staged_sol, staged_rhs, tol_rel, tol_abs);
        stage_clear();
        return res;
    }
    void get_grad_solution_staged(int loc) {
        mlmg->getGradSolution(staged_face_vec,
                              static_cast<amrex::MLLinOp::Location>(loc));
        stage_clear();
    }
    void get_fluxes_face_staged(int loc) {
        mlmg->getFluxes(staged_face_vec,
                        static_cast<amrex::MLLinOp::Location>(loc));
        stage_clear();
    }
    void get_fluxes_cc_staged(int loc) {
        mlmg->getFluxes(staged_out,
                        static_cast<amrex::MLLinOp::Location>(loc));
        stage_clear();
    }
    void comp_residual_staged() {
        mlmg->compResidual(staged_out, staged_sol, staged_rhs);
        stage_clear();
    }
    void apply_staged() {
        mlmg->apply(staged_out, staged_sol);
        stage_clear();
    }
};

struct PlotfileStage {
    std::vector<const amrex::MultiFab*> mfs;
    std::vector<amrex::Geometry> geoms;
    std::vector<int> steps;
    std::vector<amrex::IntVect> ref_ratios;

    void clear() {
        mfs.clear();
        geoms.clear();
        steps.clear();
        ref_ratios.clear();
    }
};

PlotfileStage g_plotfile_stage;

// ============================================================================
// JLCXX Module Registration
// ============================================================================

JLCXX_MODULE define_julia_module(jlcxx::Module& mod) {
    // Lifecycle & ParallelDescriptor
    mod.method("amrex_init", [](bool use_parmparse) {
        if (!amrex::Initialized()) {
            const int sigs[] = {SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGABRT, SIGINT, SIGTERM};
            struct sigaction saved_sa[7];
            for (int i = 0; i < 7; ++i) {
                sigaction(sigs[i], nullptr, &saved_sa[i]);
            }
            int argc = 0;
            char** argv = nullptr;
            amrex::Initialize(argc, argv, use_parmparse, MPI_COMM_WORLD, []() {
                amrex::system::signal_handling = false;
                amrex::system::handle_sigsegv = false;
                amrex::system::handle_sigterm = false;
                amrex::system::handle_sigint = false;
                amrex::system::handle_sigabrt = false;
                amrex::system::handle_sigfpe = false;
                amrex::system::handle_sigill = false;
            });
            for (int i = 0; i < 7; ++i) {
                sigaction(sigs[i], &saved_sa[i], nullptr);
            }
        }
        amrex::MFIter::allowMultipleMFIters(true);
    });
    mod.method("amrex_initialized", []() { return amrex::Initialized(); });
    mod.method("amrex_finalize", []() {
        if (amrex::Initialized()) {
            const int sigs[] = {SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGABRT, SIGINT, SIGTERM};
            struct sigaction saved_sa[7];
            for (int i = 0; i < 7; ++i) {
                sigaction(sigs[i], nullptr, &saved_sa[i]);
            }
            amrex::Finalize();
            for (int i = 0; i < 7; ++i) {
                sigaction(sigs[i], &saved_sa[i], nullptr);
            }
        }
    });
    mod.method("parallel_myproc", []() { return amrex::ParallelDescriptor::MyProc(); });
    mod.method("parallel_nprocs", []() { return amrex::ParallelDescriptor::NProcs(); });
    mod.method("parallel_ioprocessor", []() { return amrex::ParallelDescriptor::IOProcessor(); });
    mod.method("parallel_barrier", []() { amrex::ParallelDescriptor::Barrier(); });

    mod.method("parallel_reduce_sum_real", [](double val) {
        amrex::ParallelDescriptor::ReduceRealSum(val);
        return val;
    });
    mod.method("parallel_reduce_max_real", [](double val) {
        amrex::ParallelDescriptor::ReduceRealMax(val);
        return val;
    });
    mod.method("parallel_reduce_min_real", [](double val) {
        amrex::ParallelDescriptor::ReduceRealMin(val);
        return val;
    });
    mod.method("parallel_reduce_sum_long", [](int64_t val) {
        amrex::Long lval = val;
        amrex::ParallelDescriptor::ReduceLongSum(lval);
        return static_cast<int64_t>(lval);
    });
    mod.method("parallel_reduce_sum_real_vec!", [](jlcxx::ArrayRef<double> arr) {
        amrex::ParallelDescriptor::ReduceRealSum(arr.data(), arr.size());
    });

    // JlBox
    mod.add_type<JlBox>("JlBox")
        .constructor<>()
        .constructor<int, int, int, int, int, int>()
        .constructor<int, int, int, int, int, int, int, int, int>()
        .method("lo", &JlBox::lo)
        .method("hi", &JlBox::hi)
        .method("length", &JlBox::length)
        .method("is_nodal", &JlBox::is_nodal)
        .method("num_pts", &JlBox::num_pts)
        .method("is_empty", &JlBox::is_empty)
        .method("contains", &JlBox::contains)
        .method("surrounding_nodes", &JlBox::surrounding_nodes)
        .method("enclosed_cells", &JlBox::enclosed_cells)
        .method("grow", &JlBox::grow)
        .method("intersect", &JlBox::intersect);

    // JlRealBox
    mod.add_type<JlRealBox>("JlRealBox")
        .constructor<>()
        .constructor<double, double, double, double, double, double>()
        .method("lo", &JlRealBox::lo)
        .method("hi", &JlRealBox::hi)
        .method("length", &JlRealBox::length)
        .method("contains", &JlRealBox::contains);

    // JlGeometry
    mod.add_type<JlGeometry>("JlGeometry")
        .constructor<>()
        .constructor<const JlBox&, const JlRealBox&, int, bool, bool, bool>()
        .method("cell_size", &JlGeometry::cell_size)
        .method("inv_cell_size", &JlGeometry::inv_cell_size)
        .method("prob_lo", &JlGeometry::prob_lo)
        .method("prob_hi", &JlGeometry::prob_hi)
        .method("domain", &JlGeometry::domain)
        .method("prob_domain", &JlGeometry::prob_domain)
        .method("is_periodic", &JlGeometry::is_periodic);

    // JlBoxArray
    mod.add_type<JlBoxArray>("JlBoxArray")
        .constructor<>()
        .constructor<const JlBox&>()
        .method("max_size!", &JlBoxArray::max_size)
        .method("convert_to_nodal", &JlBoxArray::convert_to_nodal)
        .method("size", &JlBoxArray::size)
        .method("empty", &JlBoxArray::empty)
        .method("get_box", &JlBoxArray::get_box)
        .method("contains_cell", &JlBoxArray::contains_cell);

    // JlDistributionMapping
    mod.add_type<JlDistributionMapping>("JlDistributionMapping")
        .constructor<>()
        .constructor<const JlBoxArray&>()
        .method("size", &JlDistributionMapping::size)
        .method("get_rank", &JlDistributionMapping::get_rank);

    // JlMultiFab
    mod.add_type<JlMultiFab>("JlMultiFab")
        .constructor<>()
        .constructor<const JlBoxArray&, const JlDistributionMapping&, int, int>()
        .constructor<const JlBoxArray&, const JlDistributionMapping&, int, int, int, int>()
        .method("n_comp", &JlMultiFab::n_comp)
        .method("n_grow", &JlMultiFab::n_grow)
        .method("size", &JlMultiFab::size)
        .method("local_size", &JlMultiFab::local_size)
        .method("box_array", &JlMultiFab::box_array)
        .method("dmap", &JlMultiFab::dmap)
        .method("set_val!", &JlMultiFab::set_val)
        .method("set_val_comp!", &JlMultiFab::set_val_comp)
        .method("plus_val!", &JlMultiFab::plus_val)
        .method("mult_val!", &JlMultiFab::mult_val)
        .method("add_from!", &JlMultiFab::add_from)
        .method("copy_from!", &JlMultiFab::copy_from)
        .method("parallel_copy_from!", &JlMultiFab::parallel_copy_from)
        .method("fill_boundary!", &JlMultiFab::fill_boundary)
        .method("fill_boundary_comp!", &JlMultiFab::fill_boundary_comp)
        .method("sum_boundary!", &JlMultiFab::sum_boundary)
        .method("sum_boundary_comp!", &JlMultiFab::sum_boundary_comp)
        .method("min_val", &JlMultiFab::min_val)
        .method("max_val", &JlMultiFab::max_val)
        .method("norm0", &JlMultiFab::norm0)
        .method("norm1", &JlMultiFab::norm1)
        .method("norm2", &JlMultiFab::norm2)
        .method("sum_val", &JlMultiFab::sum_val);

    // JliMultiFab
    mod.add_type<JliMultiFab>("JliMultiFab")
        .constructor<>()
        .constructor<const JlBoxArray&, const JlDistributionMapping&, int, int>()
        .constructor<const JlBoxArray&, const JlDistributionMapping&, int, int, int, int>()
        .method("n_comp", &JliMultiFab::n_comp)
        .method("n_grow", &JliMultiFab::n_grow)
        .method("size", &JliMultiFab::size)
        .method("local_size", &JliMultiFab::local_size)
        .method("box_array", &JliMultiFab::box_array)
        .method("dmap", &JliMultiFab::dmap)
        .method("set_val!", &JliMultiFab::set_val)
        .method("copy_from!", &JliMultiFab::copy_from)
        .method("fill_boundary!", &JliMultiFab::fill_boundary)
        .method("sum_boundary!", &JliMultiFab::sum_boundary)
        .method("min_val", &JliMultiFab::min_val)
        .method("max_val", &JliMultiFab::max_val);

    // JlTagBoxArray & JlBCRec
    mod.add_type<JlTagBoxArray>("JlTagBoxArray")
        .constructor<>()
        .constructor<const JlBoxArray&, const JlDistributionMapping&, int>()
        .method("size", &JlTagBoxArray::size)
        .method("local_size", &JlTagBoxArray::local_size)
        .method("box_array", &JlTagBoxArray::box_array)
        .method("dmap", &JlTagBoxArray::dmap)
        .method("set_val!", &JlTagBoxArray::set_val);

    mod.add_type<JlBCRec>("JlBCRec")
        .constructor<>()
        .constructor<int, int, int, int, int, int>()
        .method("lo", &JlBCRec::lo)
        .method("hi", &JlBCRec::hi)
        .method("set_lo!", &JlBCRec::set_lo)
        .method("set_hi!", &JlBCRec::set_hi);

    // JlMFIter
    mod.add_type<JlMFIter>("JlMFIter")
        .constructor<const JlMultiFab&, bool>()
        .constructor<const JliMultiFab&, bool>()
        .constructor<const JlTagBoxArray&, bool>()
        .method("is_valid", &JlMFIter::is_valid)
        .method("next!", &JlMFIter::next)
        .method("close!", &JlMFIter::close)
        .method("index", &JlMFIter::index)
        .method("local_tile_index", &JlMFIter::local_tile_index)
        .method("valid_box", &JlMFIter::valid_box)
        .method("tile_box", &JlMFIter::tile_box)
        .method("fab_box", &JlMFIter::fab_box)
        .method("fab_lo", &JlMFIter::fab_lo)
        .method("fab_hi", &JlMFIter::fab_hi)
        .method("ifab_lo", &JlMFIter::ifab_lo)
        .method("ifab_hi", &JlMFIter::ifab_hi)
        .method("tfab_lo", &JlMFIter::tfab_lo)
        .method("tfab_hi", &JlMFIter::tfab_hi)
        .method("data_ptr", &JlMFIter::data_ptr)
        .method("idata_ptr", &JlMFIter::idata_ptr)
        .method("tdata_ptr", &JlMFIter::tdata_ptr);

    // JlAmrCore
    mod.add_type<JlAmrCore>("JlAmrCore")
        .constructor<const JlRealBox&, int, int, int, int, int,
                     bool, bool, bool, int, int, int, int, int, int, double>()
        .method("add_refine_box!", &JlAmrCore::add_refine_box)
        .method("clear_refine_boxes!", &JlAmrCore::clear_refine_boxes)
        .method("set_tag_mask!", &JlAmrCore::set_tag_mask)
        .method("set_tag_boxes!", &JlAmrCore::set_tag_boxes)
        .method("clear_tag_masks!", &JlAmrCore::clear_tag_masks)
        .method("ensure_base_grids!", &JlAmrCore::ensure_base_grids)
        .method("_init_from_scratch!", &JlAmrCore::init_from_scratch)
        .method("_regrid_levels!", &JlAmrCore::regrid_levels)
        .method("finest_level", &JlAmrCore::get_finest_level)
        .method("max_level", &JlAmrCore::get_max_level)
        .method("geom", &JlAmrCore::get_geom)
        .method("box_array", &JlAmrCore::get_box_array)
        .method("node_box_array", &JlAmrCore::get_node_box_array)
        .method("dmap", &JlAmrCore::get_dmap)
        .method("ref_ratio", &JlAmrCore::get_ref_ratio)
        .method("set_box_array!", &JlAmrCore::set_box_array)
        .method("set_dmap!", &JlAmrCore::set_dmap)
        .method("set_finest_level!", &JlAmrCore::set_finest_level);

    // Physical Boundary Conditions & Coarse-Fine Interpolation Utilities
    mod.method("fill_domain_boundary!", [](JlMultiFab& mf, const JlGeometry& geom,
                                           const JlBCRec& bc) {
        amrex::Vector<amrex::BCRec> bcs(mf.n_comp(), bc.bcr);
        amrex::FillDomainBoundary(*mf.mf, geom.geom, bcs);
    });

    mod.method("fill_domain_boundary_val!", [](JlMultiFab& mf, const JlGeometry& geom,
                                               const JlBCRec& bc, double time,
                                               double vlo_x, double vlo_y, double vlo_z,
                                               double vhi_x, double vhi_y, double vhi_z) {
        int ncomp = mf.n_comp();
        amrex::Vector<amrex::BCRec> bcs(ncomp, bc.bcr);
        ExtDirFillOp op{{vlo_x, vlo_y, vlo_z}, {vhi_x, vhi_y, vhi_z}};
        amrex::PhysBCFunct<amrex::GpuBndryFuncFab<ExtDirFillOp>> physbc(
            geom.geom, bcs, amrex::GpuBndryFuncFab<ExtDirFillOp>(op));
        physbc(*mf.mf, 0, ncomp, mf.mf->nGrowVect(), time, 0);
    });

    mod.method("interp_from_coarse_to_fine!", [](JlMultiFab& coarse, JlMultiFab& fine,
                                                 int icomp, int ncomp,
                                                 int rx, int ry, int rz,
                                                 const JlGeometry& cgeom,
                                                 const JlGeometry& fgeom,
                                                 int interp_type, int nghost) {
        amrex::MultiFab f(*fine.mf, amrex::make_alias, icomp, ncomp);
        amrex::MultiFab c(*coarse.mf, amrex::make_alias, icomp, ncomp);
        int lo_bc[3] = {amrex::BCType::int_dir, amrex::BCType::int_dir, amrex::BCType::int_dir};
        int hi_bc[3] = {amrex::BCType::int_dir, amrex::BCType::int_dir, amrex::BCType::int_dir};
        amrex::Vector<amrex::BCRec> bcs(ncomp, amrex::BCRec(lo_bc, hi_bc));
        PhysBCFunctNoOpFab<amrex::FArrayBox> cphysbc, fphysbc;
        amrex::Interpolater* mapper = get_interpolater(interp_type);
        amrex::InterpFromCoarseLevel(f, amrex::IntVect(nghost), 0.0, c, 0, 0, ncomp,
                                     cgeom.geom, fgeom.geom, cphysbc, 0, fphysbc, 0,
                                     amrex::IntVect(rx, ry, rz), mapper, bcs, 0);
    });

    mod.method("interp_from_coarse_to_fine_bc!", [](JlMultiFab& coarse, JlMultiFab& fine,
                                                    int icomp, int ncomp,
                                                    int rx, int ry, int rz,
                                                    const JlGeometry& cgeom,
                                                    const JlGeometry& fgeom,
                                                    int interp_type, int nghost,
                                                    const JlBCRec& bc,
                                                    double vlo_x, double vlo_y, double vlo_z,
                                                    double vhi_x, double vhi_y, double vhi_z) {
        amrex::MultiFab f(*fine.mf, amrex::make_alias, icomp, ncomp);
        amrex::MultiFab c(*coarse.mf, amrex::make_alias, icomp, ncomp);
        amrex::Vector<amrex::BCRec> bcs(ncomp, bc.bcr);
        ExtDirFillOp op{{vlo_x, vlo_y, vlo_z}, {vhi_x, vhi_y, vhi_z}};
        amrex::PhysBCFunct<amrex::GpuBndryFuncFab<ExtDirFillOp>> cphysbc(
            cgeom.geom, bcs, amrex::GpuBndryFuncFab<ExtDirFillOp>(op));
        amrex::PhysBCFunct<amrex::GpuBndryFuncFab<ExtDirFillOp>> fphysbc(
            fgeom.geom, bcs, amrex::GpuBndryFuncFab<ExtDirFillOp>(op));
        amrex::Interpolater* mapper = get_interpolater(interp_type);
        amrex::InterpFromCoarseLevel(f, amrex::IntVect(nghost), 0.0, c, 0, 0, ncomp,
                                     cgeom.geom, fgeom.geom, cphysbc, 0, fphysbc, 0,
                                     amrex::IntVect(rx, ry, rz), mapper, bcs, 0);
    });

    mod.method("fill_patch_single_level!", [](JlMultiFab& dst, double time,
                                              JlMultiFab& src, int scomp, int dcomp, int ncomp,
                                              const JlGeometry& geom) {
        PhysBCFunctNoOpFab<amrex::FArrayBox> physbc;
        amrex::Vector<amrex::MultiFab*> smf{src.mf.get()};
        amrex::Vector<amrex::Real> stime{time};
        amrex::FillPatchSingleLevel(*dst.mf, time, smf, stime, scomp, dcomp, ncomp,
                                    geom.geom, physbc, 0);
    });

    mod.method("fill_patch_single_level_bc!", [](JlMultiFab& dst, double time,
                                                 JlMultiFab& src, int scomp, int dcomp, int ncomp,
                                                 const JlGeometry& geom, const JlBCRec& bc,
                                                 double vlo_x, double vlo_y, double vlo_z,
                                                 double vhi_x, double vhi_y, double vhi_z) {
        amrex::Vector<amrex::BCRec> bcs(ncomp, bc.bcr);
        ExtDirFillOp op{{vlo_x, vlo_y, vlo_z}, {vhi_x, vhi_y, vhi_z}};
        amrex::PhysBCFunct<amrex::GpuBndryFuncFab<ExtDirFillOp>> physbc(
            geom.geom, bcs, amrex::GpuBndryFuncFab<ExtDirFillOp>(op));
        amrex::Vector<amrex::MultiFab*> smf{src.mf.get()};
        amrex::Vector<amrex::Real> stime{time};
        amrex::FillPatchSingleLevel(*dst.mf, time, smf, stime, scomp, dcomp, ncomp,
                                    geom.geom, physbc, 0);
    });

    mod.method("fill_patch_two_levels!", [](JlMultiFab& dst, double time,
                                            JlMultiFab& cmf, JlMultiFab& fmf,
                                            int scomp, int dcomp, int ncomp,
                                            const JlGeometry& cgeom, const JlGeometry& fgeom,
                                            int rx, int ry, int rz, int interp_type) {
        PhysBCFunctNoOpFab<amrex::FArrayBox> cphysbc, fphysbc;
        amrex::Vector<amrex::MultiFab*> cmf_vec{cmf.mf.get()};
        amrex::Vector<amrex::MultiFab*> fmf_vec{fmf.mf.get()};
        amrex::Vector<amrex::Real> ct{time}, ft{time};
        int lo_bc[3] = {amrex::BCType::int_dir, amrex::BCType::int_dir, amrex::BCType::int_dir};
        int hi_bc[3] = {amrex::BCType::int_dir, amrex::BCType::int_dir, amrex::BCType::int_dir};
        amrex::Vector<amrex::BCRec> bcs(ncomp, amrex::BCRec(lo_bc, hi_bc));
        amrex::Interpolater* mapper = get_interpolater(interp_type);
        amrex::FillPatchTwoLevels(*dst.mf, time, cmf_vec, ct, fmf_vec, ft,
                                  scomp, dcomp, ncomp, cgeom.geom, fgeom.geom,
                                  cphysbc, 0, fphysbc, 0,
                                  amrex::IntVect(rx, ry, rz), mapper, bcs, 0);
    });

    mod.method("fill_patch_two_levels_bc!", [](JlMultiFab& dst, double time,
                                               JlMultiFab& cmf, JlMultiFab& fmf,
                                               int scomp, int dcomp, int ncomp,
                                               const JlGeometry& cgeom, const JlGeometry& fgeom,
                                               int rx, int ry, int rz, int interp_type,
                                               const JlBCRec& bc,
                                               double vlo_x, double vlo_y, double vlo_z,
                                               double vhi_x, double vhi_y, double vhi_z) {
        amrex::Vector<amrex::BCRec> bcs(ncomp, bc.bcr);
        ExtDirFillOp op{{vlo_x, vlo_y, vlo_z}, {vhi_x, vhi_y, vhi_z}};
        amrex::PhysBCFunct<amrex::GpuBndryFuncFab<ExtDirFillOp>> cphysbc(
            cgeom.geom, bcs, amrex::GpuBndryFuncFab<ExtDirFillOp>(op));
        amrex::PhysBCFunct<amrex::GpuBndryFuncFab<ExtDirFillOp>> fphysbc(
            fgeom.geom, bcs, amrex::GpuBndryFuncFab<ExtDirFillOp>(op));
        amrex::Vector<amrex::MultiFab*> cmf_vec{cmf.mf.get()};
        amrex::Vector<amrex::MultiFab*> fmf_vec{fmf.mf.get()};
        amrex::Vector<amrex::Real> ct{time}, ft{time};
        amrex::Interpolater* mapper = get_interpolater(interp_type);
        amrex::FillPatchTwoLevels(*dst.mf, time, cmf_vec, ct, fmf_vec, ft,
                                  scomp, dcomp, ncomp, cgeom.geom, fgeom.geom,
                                  cphysbc, 0, fphysbc, 0,
                                  amrex::IntVect(rx, ry, rz), mapper, bcs, 0);
    });

    mod.method("average_down!", [](const JlMultiFab& fine, JlMultiFab& coarse,
                                   const JlGeometry& fgeom, const JlGeometry& cgeom,
                                   int scomp, int ncomp, int rx, int ry, int rz) {
        amrex::average_down(*fine.mf, *coarse.mf, fgeom.geom, cgeom.geom,
                            scomp, ncomp, amrex::IntVect(rx, ry, rz));
    });

    mod.method("average_down_nodal!", [](const JlMultiFab& fine, JlMultiFab& coarse,
                                         int rx, int ry, int rz, int nghost) {
        amrex::average_down_nodal(*fine.mf, *coarse.mf,
                                  amrex::IntVect(rx, ry, rz), nghost);
    });

    mod.method("average_down_faces!", [](const JlMultiFab& fx_fine, const JlMultiFab& fy_fine,
                                         const JlMultiFab& fz_fine,
                                         JlMultiFab& fx_crse, JlMultiFab& fy_crse,
                                         JlMultiFab& fz_crse,
                                         int rx, int ry, int rz, const JlGeometry& cgeom) {
        amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM> fine_arr{
            fx_fine.mf.get(), fy_fine.mf.get(), fz_fine.mf.get()
        };
        amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM> crse_arr{
            fx_crse.mf.get(), fy_crse.mf.get(), fz_crse.mf.get()
        };
        amrex::average_down_faces(fine_arr, crse_arr, amrex::IntVect(rx, ry, rz), cgeom.geom);
    });

    mod.method("average_down_face!", [](const JlMultiFab& fine, JlMultiFab& crse,
                                        int rx, int ry, int rz, const JlGeometry& cgeom) {
        amrex::average_down_faces(*fine.mf, *crse.mf, amrex::IntVect(rx, ry, rz), cgeom.geom);
    });

    mod.method("average_down_edges!", [](const JlMultiFab& ex_fine, const JlMultiFab& ey_fine,
                                         const JlMultiFab& ez_fine,
                                         JlMultiFab& ex_crse, JlMultiFab& ey_crse,
                                         JlMultiFab& ez_crse,
                                         int rx, int ry, int rz, int ngcrse) {
        amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM> fine_arr{
            ex_fine.mf.get(), ey_fine.mf.get(), ez_fine.mf.get()
        };
        amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM> crse_arr{
            ex_crse.mf.get(), ey_crse.mf.get(), ez_crse.mf.get()
        };
        amrex::average_down_edges(fine_arr, crse_arr, amrex::IntVect(rx, ry, rz), ngcrse);
    });

    mod.add_type<JlYAFluxRegister>("JlYAFluxRegister")
        .constructor<>()
        .constructor<const JlBoxArray&, const JlBoxArray&,
                     const JlDistributionMapping&, const JlDistributionMapping&,
                     const JlGeometry&, const JlGeometry&,
                     int, int, int, int, int>()
        .method("reset!", &JlYAFluxRegister::reset)
        .method("crse_add!", &JlYAFluxRegister::crse_add)
        .method("fine_add!", &JlYAFluxRegister::fine_add)
        .method("reflux!", &JlYAFluxRegister::reflux)
        .method("crse_has_work", &JlYAFluxRegister::crse_has_work)
        .method("fine_has_work", &JlYAFluxRegister::fine_has_work);

    mod.add_type<JlEdgeFluxRegister>("JlEdgeFluxRegister")
        .constructor<>()
        .constructor<const JlBoxArray&, const JlBoxArray&,
                     const JlDistributionMapping&, const JlDistributionMapping&,
                     const JlGeometry&, const JlGeometry&, int>()
        .method("reset!", &JlEdgeFluxRegister::reset)
        .method("crse_add!", &JlEdgeFluxRegister::crse_add)
        .method("fine_add!", &JlEdgeFluxRegister::fine_add)
        .method("reflux!", &JlEdgeFluxRegister::reflux);

    // Linear Solvers (LPInfo, MLPoisson, MLABecLaplacian, MLNodeLaplacian, MLMG)
    mod.add_type<JlLPInfo>("JlLPInfo")
        .constructor<>()
        .constructor<int, bool, bool, int, int, bool, int>()
        .method("set_max_coarsening_level!", &JlLPInfo::set_max_coarsening_level)
        .method("set_agglomeration!", &JlLPInfo::set_agglomeration)
        .method("set_consolidation!", &JlLPInfo::set_consolidation)
        .method("set_agg_grid_size!", &JlLPInfo::set_agg_grid_size)
        .method("set_con_grid_size!", &JlLPInfo::set_con_grid_size)
        .method("set_metric_term!", &JlLPInfo::set_metric_term)
        .method("set_hidden_direction!", &JlLPInfo::set_hidden_direction);

    mod.add_type<JlMLPoisson>("JlMLPoisson")
        .constructor<>()
        .constructor<const JlGeometry&, const JlBoxArray&, const JlDistributionMapping&, const JlLPInfo&>()
        .method("stage_clear!", &JlMLPoisson::stage_clear)
        .method("stage_push_level!", &JlMLPoisson::stage_push_level)
        .method("define_staged!", &JlMLPoisson::define_staged)
        .method("set_domain_bc!", &JlMLPoisson::set_domain_bc)
        .method("set_domain_bc_loc!", &JlMLPoisson::set_domain_bc_loc)
        .method("set_level_bc_homog!", &JlMLPoisson::set_level_bc_homog)
        .method("set_level_bc!", &JlMLPoisson::set_level_bc)
        .method("set_level_bc_robin!", &JlMLPoisson::set_level_bc_robin)
        .method("set_coarse_fine_bc!", &JlMLPoisson::set_coarse_fine_bc)
        .method("set_coarse_fine_bc_homog!", &JlMLPoisson::set_coarse_fine_bc_homog)
        .method("set_max_order!", &JlMLPoisson::set_max_order)
        .method("get_max_order", &JlMLPoisson::get_max_order)
        .method("set_verbose!", &JlMLPoisson::set_verbose)
        .method("set_enforce_singular_solvable!", &JlMLPoisson::set_enforce_singular_solvable);

    mod.add_type<JlMLABecLaplacian>("JlMLABecLaplacian")
        .constructor<>()
        .constructor<const JlGeometry&, const JlBoxArray&, const JlDistributionMapping&, const JlLPInfo&, int>()
        .method("stage_clear!", &JlMLABecLaplacian::stage_clear)
        .method("stage_push_level!", &JlMLABecLaplacian::stage_push_level)
        .method("define_staged!", &JlMLABecLaplacian::define_staged)
        .method("set_domain_bc!", &JlMLABecLaplacian::set_domain_bc)
        .method("set_domain_bc_loc!", &JlMLABecLaplacian::set_domain_bc_loc)
        .method("set_level_bc_homog!", &JlMLABecLaplacian::set_level_bc_homog)
        .method("set_level_bc!", &JlMLABecLaplacian::set_level_bc)
        .method("set_level_bc_robin!", &JlMLABecLaplacian::set_level_bc_robin)
        .method("set_coarse_fine_bc!", &JlMLABecLaplacian::set_coarse_fine_bc)
        .method("set_coarse_fine_bc_homog!", &JlMLABecLaplacian::set_coarse_fine_bc_homog)
        .method("set_max_order!", &JlMLABecLaplacian::set_max_order)
        .method("get_max_order", &JlMLABecLaplacian::get_max_order)
        .method("set_verbose!", &JlMLABecLaplacian::set_verbose)
        .method("set_enforce_singular_solvable!", &JlMLABecLaplacian::set_enforce_singular_solvable)
        .method("set_scalars!", &JlMLABecLaplacian::set_scalars)
        .method("set_a_coeffs_scalar!", &JlMLABecLaplacian::set_a_coeffs_scalar)
        .method("set_a_coeffs!", &JlMLABecLaplacian::set_a_coeffs)
        .method("set_b_coeffs_scalar!", &JlMLABecLaplacian::set_b_coeffs_scalar)
        .method("set_b_coeffs!", &JlMLABecLaplacian::set_b_coeffs);

    mod.add_type<JlMLNodeLaplacian>("JlMLNodeLaplacian")
        .constructor<>()
        .constructor<const JlGeometry&, const JlBoxArray&, const JlDistributionMapping&, const JlLPInfo&, double>()
        .method("stage_clear!", &JlMLNodeLaplacian::stage_clear)
        .method("stage_push_level!", &JlMLNodeLaplacian::stage_push_level)
        .method("define_staged!", &JlMLNodeLaplacian::define_staged)
        .method("set_domain_bc!", &JlMLNodeLaplacian::set_domain_bc)
        .method("set_max_order!", &JlMLNodeLaplacian::set_max_order)
        .method("get_max_order", &JlMLNodeLaplacian::get_max_order)
        .method("set_verbose!", &JlMLNodeLaplacian::set_verbose)
        .method("set_enforce_singular_solvable!", &JlMLNodeLaplacian::set_enforce_singular_solvable)
        .method("set_sigma!", &JlMLNodeLaplacian::set_sigma)
        .method("set_gauss_seidel!", &JlMLNodeLaplacian::set_gauss_seidel)
        .method("set_harmonic_average!", &JlMLNodeLaplacian::set_harmonic_average)
        .method("comp_divergence_1!", &JlMLNodeLaplacian::comp_divergence_1)
        .method("update_velocity_1!", &JlMLNodeLaplacian::update_velocity_1);

    mod.add_type<JlMLMG>("JlMLMG")
        .constructor<JlMLPoisson&>()
        .constructor<JlMLABecLaplacian&>()
        .constructor<JlMLNodeLaplacian&>()
        .method("set_verbose!", &JlMLMG::set_verbose)
        .method("set_bottom_verbose!", &JlMLMG::set_bottom_verbose)
        .method("set_max_iter!", &JlMLMG::set_max_iter)
        .method("set_max_fmg_iter!", &JlMLMG::set_max_fmg_iter)
        .method("set_fixed_iter!", &JlMLMG::set_fixed_iter)
        .method("set_pre_smooth!", &JlMLMG::set_pre_smooth)
        .method("set_post_smooth!", &JlMLMG::set_post_smooth)
        .method("set_final_smooth!", &JlMLMG::set_final_smooth)
        .method("set_bottom_smooth!", &JlMLMG::set_bottom_smooth)
        .method("set_bottom_solver!", &JlMLMG::set_bottom_solver)
        .method("set_bottom_max_iter!", &JlMLMG::set_bottom_max_iter)
        .method("set_bottom_tolerance!", &JlMLMG::set_bottom_tolerance)
        .method("set_bottom_tolerance_abs!", &JlMLMG::set_bottom_tolerance_abs)
        .method("set_final_fill_bc!", &JlMLMG::set_final_fill_bc)
        .method("num_amr_levels", &JlMLMG::num_amr_levels)
        .method("get_init_residual", &JlMLMG::get_init_residual)
        .method("get_final_residual", &JlMLMG::get_final_residual)
        .method("get_init_rhs", &JlMLMG::get_init_rhs)
        .method("stage_clear!", &JlMLMG::stage_clear)
        .method("stage_push_sol_rhs!", &JlMLMG::stage_push_sol_rhs)
        .method("stage_push_out_sol_rhs!", &JlMLMG::stage_push_out_sol_rhs)
        .method("stage_push_out_in!", &JlMLMG::stage_push_out_in)
        .method("stage_push_face_vec!", &JlMLMG::stage_push_face_vec)
        .method("stage_push_out!", &JlMLMG::stage_push_out)
        .method("solve_staged!", &JlMLMG::solve_staged)
        .method("get_grad_solution_staged!", &JlMLMG::get_grad_solution_staged)
        .method("get_fluxes_face_staged!", &JlMLMG::get_fluxes_face_staged)
        .method("get_fluxes_cc_staged!", &JlMLMG::get_fluxes_cc_staged)
        .method("comp_residual_staged!", &JlMLMG::comp_residual_staged)
        .method("apply_staged!", &JlMLMG::apply_staged);

    mod.method("sum_fine_to_coarse_nodal!", [](JlMultiFab& coarse, const JlMultiFab& fine,
                                               int icomp, int ncomp,
                                               int rx, int ry, int rz) {
        amrex::MultiFab f(*fine.mf, amrex::make_alias, icomp, ncomp);
        amrex::MultiFab c(*coarse.mf, amrex::make_alias, icomp, ncomp);
        amrex::MultiFab ctmp(c.boxArray(), c.DistributionMap(), ncomp, 0);
        ctmp.setVal(0.0);
        amrex::average_down_nodal(f, ctmp, amrex::IntVect(rx, ry, rz));
        amrex::MultiFab::Add(c, ctmp, 0, 0, ncomp, 0);
    });

    // JlParticleContainer & JlParIter
    mod.add_type<JlParticleContainer>("JlParticleContainer")
        .constructor<JlAmrCore&, int, int>()
        .constructor<const JlGeometry&, const JlDistributionMapping&, const JlBoxArray&, int, int>()
        .method("reserve_and_resize_data!", &JlParticleContainer::reserve_and_resize_data)
        .method("ensure_tile!", &JlParticleContainer::ensure_tile)
        .method("resize_tile!", &JlParticleContainer::resize_tile)
        .method("tile_num_particles", &JlParticleContainer::tile_num_particles)
        .method("tile_idcpu_ptr", &JlParticleContainer::tile_idcpu_ptr)
        .method("tile_pos_ptr", &JlParticleContainer::tile_pos_ptr)
        .method("tile_real_ptr", &JlParticleContainer::tile_real_ptr)
        .method("tile_int_ptr", &JlParticleContainer::tile_int_ptr)
        .method("redistribute!", &JlParticleContainer::redistribute)
        .method("total_number_of_particles", &JlParticleContainer::total_number_of_particles)
        .method("number_of_particles_at_level", &JlParticleContainer::number_of_particles_at_level)
        .method("clear_particles!", &JlParticleContainer::clear_particles)
        .method("write_plotfile", &JlParticleContainer::write_plotfile)
        .method("checkpoint", &JlParticleContainer::checkpoint)
        .method("restart!", &JlParticleContainer::restart)
        .method("write_plotfile_hdf5", &JlParticleContainer::write_plotfile_hdf5)
        .method("checkpoint_hdf5", &JlParticleContainer::checkpoint_hdf5)
        .method("restart_hdf5!", &JlParticleContainer::restart_hdf5);

    mod.add_type<JlParIter>("JlParIter")
        .constructor<JlParticleContainer&, int>()
        .method("is_valid", &JlParIter::is_valid)
        .method("next!", &JlParIter::next)
        .method("close!", &JlParIter::close)
        .method("index", &JlParIter::index)
        .method("local_tile_index", &JlParIter::local_tile_index)
        .method("valid_box", &JlParIter::valid_box)
        .method("tile_box", &JlParIter::tile_box)
        .method("num_particles", &JlParIter::num_particles)
        .method("resize_tile!", &JlParIter::resize_tile)
        .method("idcpu_ptr", &JlParIter::idcpu_ptr)
        .method("pos_ptr", &JlParIter::pos_ptr)
        .method("real_ptr", &JlParIter::real_ptr)
        .method("int_ptr", &JlParIter::int_ptr);

    mod.method("next_particle_id", []() {
        return static_cast<int64_t>(amrex::Particle<0, 0>::NextID());
    });

    // Plotfile & VisMF I/O
    mod.method("write_single_level_plotfile", [](const std::string& plotfilename,
                                                 const JlMultiFab& mf,
                                                 const std::string& varnames_csv,
                                                 const JlGeometry& geom,
                                                 double time, int level_step) {
        auto vnames = split_varnames(varnames_csv, mf.n_comp());
        amrex::Vector<std::string> amrex_vnames(vnames.begin(), vnames.end());
        amrex::WriteSingleLevelPlotfile(plotfilename, *mf.mf, amrex_vnames,
                                        geom.geom, time, level_step);
    });

    mod.method("write_two_level_plotfile", [](const std::string& plotfilename,
                                              const JlMultiFab& mf0,
                                              const JlMultiFab& mf1,
                                              const std::string& varnames_csv,
                                              const JlGeometry& geom0,
                                              const JlGeometry& geom1,
                                              double time, int step,
                                              int rx, int ry, int rz) {
        auto vnames = split_varnames(varnames_csv, mf0.n_comp());
        amrex::Vector<std::string> amrex_vnames(vnames.begin(), vnames.end());
        amrex::Vector<const amrex::MultiFab*> mfs{mf0.mf.get(), mf1.mf.get()};
        amrex::Vector<amrex::Geometry> geoms{geom0.geom, geom1.geom};
        amrex::Vector<int> steps{step, step};
        amrex::Vector<amrex::IntVect> rr{amrex::IntVect(rx, ry, rz)};
        amrex::WriteMultiLevelPlotfile(plotfilename, 2, mfs, amrex_vnames,
                                       geoms, time, steps, rr);
    });

    mod.method("plotfile_stage_clear!", []() {
        g_plotfile_stage.clear();
    });

    mod.method("plotfile_stage_push_level!", [](const JlMultiFab& mf,
                                                const JlGeometry& geom,
                                                int step,
                                                int rx, int ry, int rz) {
        g_plotfile_stage.mfs.push_back(mf.mf.get());
        g_plotfile_stage.geoms.push_back(geom.geom);
        g_plotfile_stage.steps.push_back(step);
        if (rx > 0 && ry > 0 && rz > 0) {
            g_plotfile_stage.ref_ratios.emplace_back(rx, ry, rz);
        }
    });

    mod.method("write_staged_multilevel_plotfile", [](const std::string& plotfilename,
                                                      const std::string& varnames_csv,
                                                      double time) {
        int nlevels = static_cast<int>(g_plotfile_stage.mfs.size());
        if (nlevels <= 0) return;
        int ncomp = g_plotfile_stage.mfs[0]->nComp();
        auto vnames = split_varnames(varnames_csv, ncomp);
        amrex::Vector<std::string> amrex_vnames(vnames.begin(), vnames.end());
        amrex::Vector<const amrex::MultiFab*> mfs(g_plotfile_stage.mfs.begin(),
                                                  g_plotfile_stage.mfs.end());
        amrex::Vector<amrex::Geometry> geoms(g_plotfile_stage.geoms.begin(),
                                             g_plotfile_stage.geoms.end());
        amrex::Vector<int> steps(g_plotfile_stage.steps.begin(),
                                 g_plotfile_stage.steps.end());
        amrex::Vector<amrex::IntVect> rr(g_plotfile_stage.ref_ratios.begin(),
                                         g_plotfile_stage.ref_ratios.end());
        g_plotfile_stage.clear();
        amrex::WriteMultiLevelPlotfile(plotfilename, nlevels, mfs, amrex_vnames,
                                       geoms, time, steps, rr);
    });

    mod.method("write_single_level_plotfile_hdf5", [](const std::string& plotfilename,
                                                      const JlMultiFab& mf,
                                                      const std::string& varnames_csv,
                                                      const JlGeometry& geom,
                                                      double time, int level_step,
                                                      const std::string& compression) {
#ifdef AMREX_USE_HDF5
        auto vnames = split_varnames(varnames_csv, mf.n_comp());
        amrex::Vector<std::string> amrex_vnames(vnames.begin(), vnames.end());
        amrex::WriteSingleLevelPlotfileHDF5(plotfilename, *mf.mf, amrex_vnames,
                                            geom.geom, time, level_step, compression);
#else
        amrex::ignore_unused(plotfilename, mf, varnames_csv, geom, time, level_step, compression);
        amrex::Abort("AMReX was not built with HDF5 support (AMREX_USE_HDF5)");
#endif
    });

    mod.method("write_staged_multilevel_plotfile_hdf5", [](const std::string& plotfilename,
                                                           const std::string& varnames_csv,
                                                           double time,
                                                           const std::string& compression) {
#ifdef AMREX_USE_HDF5
        int nlevels = static_cast<int>(g_plotfile_stage.mfs.size());
        if (nlevels <= 0) return;
        int ncomp = g_plotfile_stage.mfs[0]->nComp();
        auto vnames = split_varnames(varnames_csv, ncomp);
        amrex::Vector<std::string> amrex_vnames(vnames.begin(), vnames.end());
        amrex::Vector<const amrex::MultiFab*> mfs(g_plotfile_stage.mfs.begin(),
                                                  g_plotfile_stage.mfs.end());
        amrex::Vector<amrex::Geometry> geoms(g_plotfile_stage.geoms.begin(),
                                             g_plotfile_stage.geoms.end());
        amrex::Vector<int> steps(g_plotfile_stage.steps.begin(),
                                 g_plotfile_stage.steps.end());
        amrex::Vector<amrex::IntVect> rr(g_plotfile_stage.ref_ratios.begin(),
                                         g_plotfile_stage.ref_ratios.end());
        g_plotfile_stage.clear();
        amrex::WriteMultiLevelPlotfileHDF5(plotfilename, nlevels, mfs, amrex_vnames,
                                           geoms, time, steps, rr, compression);
#else
        g_plotfile_stage.clear();
        amrex::ignore_unused(plotfilename, varnames_csv, time, compression);
        amrex::Abort("AMReX was not built with HDF5 support (AMREX_USE_HDF5)");
#endif
    });

    mod.method("vismf_write", [](const JlMultiFab& mf, const std::string& name) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            std::string d = amrex::VisMF::DirName(name);
            if (!d.empty()) {
                amrex::UtilCreateDirectory(d, 0755);
            }
        }
        amrex::ParallelDescriptor::Barrier();
        return static_cast<int64_t>(amrex::VisMF::Write(*mf.mf, name));
    });

    mod.method("vismf_read!", [](JlMultiFab& mf, const std::string& name) {
        amrex::VisMF::Read(*mf.mf, name);
    });

    mod.method("vismf_exist", [](const std::string& name) {
        return amrex::VisMF::Exist(name);
    });
}
