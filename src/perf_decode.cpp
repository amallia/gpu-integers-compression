#include "CLI/CLI.hpp"


bool read_query(std::vector<uint32_t> &ret, std::istream &is = std::cin) {
    ret.clear();
    std::string line;
    if (!std::getline(is, line))
        return false;
    std::istringstream iline(line);
    uint32_t       term_id;
    while (iline >> term_id) {
        ret.push_back(term_id);
    }
    return true;
}

int main(int argc, char const *argv[])
{
    std::string type;
    std::string index_basename;
    std::string query_basename;

    CLI::App app{"compress_index - a tool for compressing an index."};
    app.add_option("-t,--type", type, "Index type")->required();
    app.add_option("-i,--index", index_basename, "Index basename")->required();
    app.add_option("-q,--query", query_basename, "Query basename")->required();
    CLI11_PARSE(app, argc, argv);

    std::vector<uint32_t> terms;
    std::filebuf fb;
    size_t queries_num = 0;
    if (fb.open(query_basename, std::ios::in)) {
        std::istream is(&fb);
        std::vector<uint32_t> q;
        while (read_query(q, is)) {
            queries_num+=1;
            terms.insert(terms.end(), q.begin(), q.end());
        }
    }

    std::ifstream fin(index_basename, std::ios::binary);


    std::cout << "Queries: " << queries_num << ", terms: " << terms.size() << std::endl;


    return 0;
}
